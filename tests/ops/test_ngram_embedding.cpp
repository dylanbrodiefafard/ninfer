#include "ninfer/ops/ngram_embedding.h"
#include "ops/op_tester.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::uint64_t kMixIncrement = 0x9E3779B97F4A7C15ULL;

struct OracleConfig {
    std::int32_t vocabulary_size;
    std::int32_t eos_token_id;
    std::int32_t ple_layer_index;
    std::uint64_t seed;
    std::int32_t vocab_base;
};

std::uint64_t oracle_splitmix64(std::uint64_t value) {
    value += kMixIncrement;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

bool oracle_prime(std::int64_t value) {
    if (value < 2) { return false; }
    for (std::int64_t divisor = 2; divisor <= value / divisor; ++divisor) {
        if (value % divisor == 0) { return false; }
    }
    return true;
}

std::int64_t signed_from_u64(std::uint64_t value) {
    if (value <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return static_cast<std::int64_t>(value);
    }
    const std::uint64_t magnitude = (~value) + 1U;
    if (magnitude == (UINT64_C(1) << 63U)) { return std::numeric_limits<std::int64_t>::min(); }
    return -static_cast<std::int64_t>(magnitude);
}

std::int32_t oracle_remainder(std::uint64_t bits, std::int32_t modulus) {
    const std::int64_t signed_value = signed_from_u64(bits);
    std::int64_t result             = signed_value % modulus;
    if (result < 0) { result += modulus; }
    return static_cast<std::int32_t>(result);
}

struct OracleResult {
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> history;
};

OracleResult oracle(const std::vector<std::int32_t>& input,
                    const std::vector<std::int32_t>& valid,
                    const std::vector<std::int32_t>& old_history, std::int32_t width,
                    std::int32_t requests, const OracleConfig& config) {
    std::array<std::uint64_t, 3> multiplier{};
    const std::uint64_t limit =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
        static_cast<std::uint64_t>(config.vocabulary_size);
    const std::uint64_t half = limit / 2U == 0 ? 1U : limit / 2U;
    const std::uint64_t layer_seed =
        config.seed + 10007ULL * static_cast<std::uint64_t>(config.ple_layer_index);
    for (std::size_t lag = 0; lag < multiplier.size(); ++lag) {
        multiplier[lag] =
            2U * (oracle_splitmix64(layer_seed + kMixIncrement * (lag + 1U)) % half) + 1U;
    }

    std::array<std::int32_t, 16> primes{};
    std::array<std::int32_t, 16> offsets{};
    const std::int64_t skip = static_cast<std::int64_t>(config.ple_layer_index) * 16;
    std::int64_t prime_count = 0;
    std::int64_t candidate   = config.vocab_base;
    std::int32_t offset      = 0;
    while (prime_count < skip + 16) {
        if (oracle_prime(candidate)) {
            if (prime_count >= skip) {
                const auto local     = static_cast<std::size_t>(prime_count - skip);
                primes[local]        = static_cast<std::int32_t>(candidate);
                offsets[local]       = offset;
                offset             += primes[local];
            }
            ++prime_count;
        }
        ++candidate;
    }

    OracleResult result;
    result.rows.assign(static_cast<std::size_t>(16) * width * requests, -1);
    result.history.resize(static_cast<std::size_t>(2) * requests);
    for (std::int32_t request = 0; request < requests; ++request) {
        std::int32_t older = old_history[static_cast<std::size_t>(2) * request];
        std::int32_t newer = old_history[static_cast<std::size_t>(2) * request + 1];
        for (std::int32_t token = 0; token < valid[request]; ++token) {
            const std::int32_t current = input[static_cast<std::size_t>(request) * width + token];
            const std::int32_t second_previous =
                newer == config.eos_token_id ? config.eos_token_id : older;
            const std::array<std::int32_t, 3> lag{current, newer, second_previous};
            std::uint64_t mixed = 0;
            for (std::int32_t order = 2; order <= 3; ++order) {
                mixed ^= static_cast<std::uint64_t>(lag[order - 1]) * multiplier[order - 1];
                if (order == 2) {
                    mixed ^= static_cast<std::uint64_t>(lag[0]) * multiplier[0];
                }
                const std::int32_t first_head = (order - 2) * 8;
                for (std::int32_t local = 0; local < 8; ++local) {
                    const std::int32_t head = first_head + local;
                    const std::size_t index =
                        (static_cast<std::size_t>(request) * width + token) * 16 + head;
                    result.rows[index] = offsets[head] + oracle_remainder(mixed, primes[head]);
                }
            }
            older = newer;
            newer = current;
        }
        result.history[static_cast<std::size_t>(2) * request]     = older;
        result.history[static_cast<std::size_t>(2) * request + 1] = newer;
    }
    return result;
}

struct DeviceCaseResult {
    std::vector<std::int32_t> rows;
    std::vector<std::int32_t> history;
    int guard_failures;
};

DeviceCaseResult run_device(const std::vector<std::int32_t>& input,
                            const std::vector<std::int32_t>& valid,
                            const std::vector<std::int32_t>& old_history, std::int32_t width,
                            std::int32_t requests, const ops::NgramRowConfig& config,
                            bool in_place) {
    GuardedDeviceBuffer device_input(input.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_valid(valid.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_old(old_history.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_new(old_history.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_rows(static_cast<std::size_t>(16) * width * requests *
                                    sizeof(std::int32_t));
    device_input.copy_from_host(input.data(), input.size() * sizeof(std::int32_t));
    device_valid.copy_from_host(valid.data(), valid.size() * sizeof(std::int32_t));
    device_old.copy_from_host(old_history.data(), old_history.size() * sizeof(std::int32_t));
    device_new.fill(0xcd);
    device_rows.fill(0xcd);

    Tensor input_tensor(device_input.data(), DType::I32, {width, requests});
    Tensor valid_tensor(device_valid.data(), DType::I32, {requests});
    Tensor old_tensor(device_old.data(), DType::I32, {2, requests});
    Tensor new_tensor(device_new.data(), DType::I32, {2, requests});
    Tensor rows_tensor(device_rows.data(), DType::I32, {16, width, requests});
    Tensor& output_history = in_place ? old_tensor : new_tensor;
    ops::ngram_row_ids(input_tensor, valid_tensor, old_tensor, config, rows_tensor, output_history,
                       nullptr);
    cuda_synchronize();

    DeviceCaseResult result{
        from_device<std::int32_t>(device_rows.data(), static_cast<std::size_t>(16) * width * requests),
        from_device<std::int32_t>(in_place ? device_old.data() : device_new.data(),
                                  old_history.size()),
        0};
    result.guard_failures +=
        verify_exact("ngram preserves input",
                     from_device<std::int32_t>(device_input.data(), input.size()), input);
    result.guard_failures +=
        verify_exact("ngram preserves valid_tokens",
                     from_device<std::int32_t>(device_valid.data(), valid.size()), valid);
    if (!in_place) {
        result.guard_failures +=
            verify_exact("ngram preserves old_history",
                         from_device<std::int32_t>(device_old.data(), old_history.size()),
                         old_history);
    }
    result.guard_failures += device_input.verify_guards("ngram input");
    result.guard_failures += device_valid.verify_guards("ngram valid_tokens");
    result.guard_failures += device_old.verify_guards("ngram old_history");
    result.guard_failures += device_rows.verify_guards("ngram row_ids");
    if (!in_place) { result.guard_failures += device_new.verify_guards("ngram new_history"); }
    return result;
}

int known_answer_case() {
    constexpr std::int32_t eos = 248044;
    const ops::NgramRowConfig config{248320, eos, 0, 1234, 20000000};
    const std::vector<std::int32_t> input{10, 20, 30};
    const std::vector<std::int32_t> valid{3};
    const std::vector<std::int32_t> history{eos, eos};
    const std::vector<std::int32_t> expected_rows{
        6826666,   27775725,  51991156,  74082527,  82622748,  119600976, 135816374,
        152166807, 174244281, 190221032, 211723794, 232787707, 243645790, 275729718,
        280030017, 303574322, 4810669,   34962340,  40038404,  63145186,  97237011,
        115267695, 122313706, 158375242, 171840321, 191688874, 219996824, 228806855,
        252506948, 276566639, 284666446, 305764640, 9878115,   26555603,  54895210,
        62571545,  80580723,  119917398, 128922427, 147596134, 168936175, 195223391,
        219226064, 233524685, 246670267, 279816194, 297531600, 306108296};
    const std::vector<std::int32_t> expected_history{20, 30};
    const DeviceCaseResult actual = run_device(input, valid, history, 3, 1, config, false);
    int failures = verify_exact("ngram known rows", actual.rows, expected_rows);
    failures += verify_exact("ngram known history", actual.history, expected_history);
    failures += actual.guard_failures;
    return failures;
}

int continuation_case() {
    constexpr std::int32_t eos = 248044;
    const ops::NgramRowConfig config{248320, eos, 0, 1234, 20000000};
    const std::vector<std::int32_t> expected_rows{
        11251501, 34567287, 46225500, 74547382, 84537869, 101201472, 136196980, 149524337,
        178516953, 196243462, 200258135, 226017340, 244881709, 263746180, 290034520,
        301475424};
    const DeviceCaseResult actual = run_device({40}, {1}, {20, 30}, 1, 1, config, true);
    int failures = verify_exact("ngram continuation rows", actual.rows, expected_rows);
    failures += verify_exact("ngram continuation history", actual.history,
                             std::vector<std::int32_t>{30, 40});
    failures += actual.guard_failures;
    return failures;
}

int eos_reset_case() {
    constexpr std::int32_t eos = 248044;
    const ops::NgramRowConfig config{248320, eos, 0, 1234, 20000000};
    const std::vector<std::int32_t> input{7, eos, 9};
    const std::vector<std::int32_t> expected_rows{
        2927653,   34980843,  54748278,  66612378,  97814964,  109013870, 126560393,
        151352333, 167888935, 182235580, 215170017, 237467519, 247510681, 278779700,
        296141806, 304994522, 10204458,  27984170,  41283776,  68842151,  85621153,
        118821647, 129504214, 158727320, 162716417, 183296409, 205500418, 223498012,
        243883332, 265230110, 285892800, 310808036, 18043673,  37626835,  51159316,
        78294604,  94015356,  106720349, 136526052, 144330141, 176817901, 186368539,
        203707490, 230017629, 247662678, 266533413, 293096193, 307951937};
    const DeviceCaseResult actual = run_device(input, {3}, {eos, eos}, 3, 1, config, false);
    int failures = verify_exact("ngram EOS reset rows", actual.rows, expected_rows);
    failures += verify_exact("ngram EOS reset history", actual.history,
                             std::vector<std::int32_t>{eos, 9});
    failures += actual.guard_failures;
    return failures;
}

int batched_oracle_case(bool in_place) {
    constexpr std::int32_t width = 6;
    constexpr std::int32_t count = 8;
    const OracleConfig oracle_config{101, 100, 0, UINT64_C(0xfedcba9876543210), 1009};
    const ops::NgramRowConfig config{oracle_config.vocabulary_size, oracle_config.eos_token_id,
                                      oracle_config.ple_layer_index, oracle_config.seed,
                                      oracle_config.vocab_base};
    const std::vector<std::int32_t> input{
        1, 2, 3, 4, 5, 6,
        7, 100, 8, 91, 92, 93,
        10, 11, 12, 13, 14, 15,
        21, 22, 23, 24, 25, 26,
        30, 31, 32, 33, 34, 35,
        40, 100, 41, 42, 43, 44,
        50, 51, 52, 53, 54, 55,
        60, 61, 62, 63, 64, 65};
    const std::vector<std::int32_t> valid{6, 3, 0, 1, 2, 5, 4, 6};
    const std::vector<std::int32_t> history{
        99, 98, 31, 32, 41, 42, 100, 100, 70, 71, 72, 73, 74, 100, 75, 76};
    const OracleResult expected = oracle(input, valid, history, width, count, oracle_config);
    const DeviceCaseResult actual = run_device(input, valid, history, width, count, config, in_place);
    const std::string label = in_place ? "ngram batched in-place" : "ngram batched distinct";
    int failures = verify_exact((label + " rows").c_str(), actual.rows, expected.rows);
    failures += verify_exact((label + " history").c_str(), actual.history, expected.history);
    failures += actual.guard_failures;
    return failures;
}

int chained_schedule_case() {
    constexpr std::int32_t eos = 100;
    const OracleConfig oracle_config{101, eos, 0, UINT64_C(0xfedcba9876543210), 1009};
    const ops::NgramRowConfig config{oracle_config.vocabulary_size, oracle_config.eos_token_id,
                                      oracle_config.ple_layer_index, oracle_config.seed,
                                      oracle_config.vocab_base};
    // The EOS at column 2 is the final token of the first multi-token chunk. The next invocation
    // must resolve both missing predecessors from the carried raw history without rewriting it.
    const std::vector<std::int32_t> input{7, 8, eos, 9, 10, eos, 11, 12};
    const std::vector<std::int32_t> initial_history{5, 6};
    const OracleResult expected =
        oracle(input, {static_cast<std::int32_t>(input.size())}, initial_history,
               static_cast<std::int32_t>(input.size()), 1, oracle_config);
    const DeviceCaseResult one_shot =
        run_device(input, {static_cast<std::int32_t>(input.size())}, initial_history,
                   static_cast<std::int32_t>(input.size()), 1, config, false);

    auto run_schedule = [&](const std::vector<std::int32_t>& widths) {
        std::vector<std::int32_t> rows;
        std::vector<std::int32_t> history = initial_history;
        std::size_t offset                = 0;
        int guard_failures                = 0;
        for (const std::int32_t width : widths) {
            const std::vector<std::int32_t> chunk(input.begin() + offset,
                                                  input.begin() + offset + width);
            const DeviceCaseResult result =
                run_device(chunk, {width}, history, width, 1, config, false);
            rows.insert(rows.end(), result.rows.begin(), result.rows.end());
            history = result.history;
            guard_failures += result.guard_failures;
            offset += static_cast<std::size_t>(width);
        }
        if (offset != input.size()) {
            std::cerr << "FAIL ngram chained schedule did not consume the complete input\n";
            ++guard_failures;
        }
        return DeviceCaseResult{std::move(rows), std::move(history), guard_failures};
    };

    const DeviceCaseResult chunks = run_schedule({3, 2, 3});
    const DeviceCaseResult token_steps = run_schedule({1, 1, 1, 1, 1, 1, 1, 1});
    int failures = verify_exact("ngram chained one-shot rows", one_shot.rows, expected.rows);
    failures += verify_exact("ngram chained chunk rows", chunks.rows, one_shot.rows);
    failures += verify_exact("ngram chained W=1 rows", token_steps.rows, one_shot.rows);
    failures += verify_exact("ngram chained one-shot history", one_shot.history,
                             expected.history);
    failures += verify_exact("ngram chained chunk history", chunks.history, expected.history);
    failures += verify_exact("ngram chained W=1 history", token_steps.history,
                             expected.history);
    failures += verify_exact("ngram chained explicit final history", expected.history,
                             std::vector<std::int32_t>{11, 12});
    failures += one_shot.guard_failures + chunks.guard_failures + token_steps.guard_failures;
    return failures;
}

int host_step_case() {
    constexpr std::int32_t eos = 248044;
    const OracleConfig oracle_config{248320, eos, 0, 1234, 20000000};
    const ops::NgramRowConfig config{oracle_config.vocabulary_size, oracle_config.eos_token_id,
                                      oracle_config.ple_layer_index, oracle_config.seed,
                                      oracle_config.vocab_base};
    const ops::PreparedNgramRowConfig prepared = ops::prepare_ngram_row_config(config);
    const std::vector<std::int32_t> input{7, eos, 9, 10, eos, eos, 11};
    const std::vector<std::int32_t> initial_history{eos, eos};
    const OracleResult expected =
        oracle(input, {static_cast<std::int32_t>(input.size())}, initial_history,
               static_cast<std::int32_t>(input.size()), 1, oracle_config);

    std::array<std::int32_t, 2> host_history{eos, eos};
    std::vector<std::int32_t> device_history = initial_history;
    int failures = 0;
    for (std::size_t token = 0; token < input.size(); ++token) {
        const ops::NgramRowHostStep host =
            ops::ngram_row_ids_host_step(input[token], host_history, prepared);
        const DeviceCaseResult device =
            run_device({input[token]}, {1}, device_history, 1, 1, config, false);
        const auto expected_begin = expected.rows.begin() +
                                    static_cast<std::ptrdiff_t>(token * host.row_ids.size());
        const std::vector<std::int32_t> expected_rows(
            expected_begin, expected_begin + static_cast<std::ptrdiff_t>(host.row_ids.size()));
        failures += verify_exact("ngram host step oracle rows",
                                 std::vector<std::int32_t>(host.row_ids.begin(),
                                                           host.row_ids.end()),
                                 expected_rows);
        failures += verify_exact("ngram host step GPU rows",
                                 std::vector<std::int32_t>(host.row_ids.begin(),
                                                           host.row_ids.end()),
                                 device.rows);
        failures += verify_exact("ngram host step GPU history",
                                 std::vector<std::int32_t>(host.new_history.begin(),
                                                           host.new_history.end()),
                                 device.history);
        failures += device.guard_failures;
        host_history = host.new_history;
        device_history = device.history;
    }
    failures += verify_exact("ngram host step final oracle history",
                             std::vector<std::int32_t>(host_history.begin(), host_history.end()),
                             expected.history);

    const ops::NgramRowHostStep reset_first =
        ops::ngram_row_ids_host_step(input.front(), {eos, eos}, prepared);
    const ops::NgramRowHostStep replay_first =
        ops::ngram_row_ids_host_step(input.front(), {eos, eos}, prepared);
    failures += verify_exact("ngram host explicit reset rows",
                             std::vector<std::int32_t>(reset_first.row_ids.begin(),
                                                       reset_first.row_ids.end()),
                             std::vector<std::int32_t>(replay_first.row_ids.begin(),
                                                       replay_first.row_ids.end()));
    failures += verify_exact("ngram host explicit reset history",
                             std::vector<std::int32_t>(reset_first.new_history.begin(),
                                                       reset_first.new_history.end()),
                             std::vector<std::int32_t>{eos, input.front()});

    try {
        (void)ops::ngram_row_ids_host_step(248320, {eos, eos}, prepared);
        std::cerr << "FAIL ngram host accepted an out-of-vocabulary input\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        (void)ops::ngram_row_ids_host_step(7, {-1, eos}, prepared);
        std::cerr << "FAIL ngram host accepted an invalid history\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

int validation_cases() {
    const ops::NgramRowConfig good{101, 100, 0, 1234, 1009};
    GuardedDeviceBuffer input(sizeof(std::int32_t) * 2);
    GuardedDeviceBuffer valid(sizeof(std::int32_t));
    GuardedDeviceBuffer history(sizeof(std::int32_t) * 2);
    GuardedDeviceBuffer rows(sizeof(std::int32_t) * 32);
    GuardedDeviceBuffer new_history(sizeof(std::int32_t) * 2);
    Tensor input_tensor(input.data(), DType::I32, {2, 1});
    Tensor valid_tensor(valid.data(), DType::I32, {1});
    Tensor history_tensor(history.data(), DType::I32, {2, 1});
    Tensor rows_tensor(rows.data(), DType::I32, {16, 2, 1});
    Tensor new_tensor(new_history.data(), DType::I32, {2, 1});

    int failures = 0;
    auto expect_invalid = [&](const char* label, const auto& invoke) {
        try {
            invoke();
            std::cerr << "FAIL " << label << ": expected invalid_argument\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    expect_invalid("ngram bad EOS", [&] {
        const ops::NgramRowConfig bad{101, 101, 0, 1234, 1009};
        ops::ngram_row_ids(input_tensor, valid_tensor, history_tensor, bad, rows_tensor, new_tensor,
                           nullptr);
    });
    expect_invalid("ngram negative PLE module index", [&] {
        const ops::NgramRowConfig bad{101, 100, -1, 1234, 1009};
        ops::ngram_row_ids(input_tensor, valid_tensor, history_tensor, bad, rows_tensor, new_tensor,
                           nullptr);
    });
    expect_invalid("ngram second PLE module index", [&] {
        const ops::NgramRowConfig bad{101, 100, 1, 1234, 1009};
        ops::ngram_row_ids(input_tensor, valid_tensor, history_tensor, bad, rows_tensor, new_tensor,
                           nullptr);
    });
    expect_invalid("ngram row shape", [&] {
        Tensor bad_rows(rows.data(), DType::I32, {8, 2, 1});
        ops::ngram_row_ids(input_tensor, valid_tensor, history_tensor, good, bad_rows, new_tensor,
                           nullptr);
    });
    expect_invalid("ngram dtype", [&] {
        Tensor bad_input(input.data(), DType::FP32, {2, 1});
        ops::ngram_row_ids(bad_input, valid_tensor, history_tensor, good, rows_tensor, new_tensor,
                           nullptr);
    });
    expect_invalid("ngram forbidden alias", [&] {
        Tensor aliased_rows(input.data(), DType::I32, {16, 2, 1});
        ops::ngram_row_ids(input_tensor, valid_tensor, history_tensor, good, aliased_rows,
                           new_tensor, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (require_cuda() != 0) { return 1; }

    int failures = 0;
    failures += known_answer_case();
    failures += continuation_case();
    failures += eos_reset_case();
    failures += batched_oracle_case(false);
    failures += batched_oracle_case(true);
    failures += chained_schedule_case();
    failures += host_step_case();
    failures += validation_cases();
    std::cout << (failures ? "FAIL" : "OK") << " ngram_embedding\n";
    return failures ? 1 : 0;
}
