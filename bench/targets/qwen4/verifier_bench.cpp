#include "core/device.h"
#include "core/nvtx_range.h"
#include "targets/qwen4/verifier.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::int32_t kVocabulary = 248320;
constexpr std::array<std::int32_t, 4> kDefaultInputs = {48, 16451, 17120, 22188};
constexpr std::array<std::int32_t, 4> kDefaultTargets = {16451, 17120, 22188, 11988};

struct Options {
    std::filesystem::path weights;
    std::vector<std::int32_t> inputs{kDefaultInputs.begin(), kDefaultInputs.end()};
    std::vector<std::int32_t> targets{kDefaultTargets.begin(), kDefaultTargets.end()};
    int device      = 0;
    int warmup      = 1;
    int repetitions = 3;
    bool profile    = false;
};

void print_usage(const char* executable) {
    std::cout
        << "usage: " << executable
        << " --weights <qwen4-verifier.ninfer> [--device <id>] [--warmup <sequences>]"
           " [--repetitions <sequences>] [--inputs <id,id,...>]"
           " [--targets <id,id,...>] [--profile]\n\n"
           "The artifact path is mandatory. QSA KV is always the verifier's intrinsic"
           " NVFP4-G16 layout. --profile requires --repetitions 1 and brackets only the"
           " measured sequence with the CUDA profiler API.\n";
}

std::vector<std::int32_t> parse_ids(std::string_view text, std::string_view option) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        if (end == begin) {
            throw std::invalid_argument(std::string(option) + " contains an empty token id");
        }
        std::int32_t value = 0;
        const char* first = text.data() + begin;
        const char* last = text.data() + end;
        const auto [parsed, error] = std::from_chars(first, last, value);
        if (error != std::errc{} || parsed != last) {
            throw std::invalid_argument(std::string(option) + " contains a non-integer token id");
        }
        result.push_back(value);
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    if (result.empty()) { throw std::invalid_argument(std::string(option) + " must not be empty"); }
    return result;
}

int parse_int(std::string_view text, std::string_view option) {
    int value = 0;
    const auto [parsed, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || parsed != text.data() + text.size()) {
        throw std::invalid_argument(std::string(option) + " requires an integer");
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool weights_set = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](std::string_view name) -> std::string_view {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--weights") {
            options.weights = std::string(value(argument));
            weights_set = true;
        } else if (argument == "--device") {
            options.device = parse_int(value(argument), argument);
        } else if (argument == "--warmup") {
            options.warmup = parse_int(value(argument), argument);
        } else if (argument == "--repetitions") {
            options.repetitions = parse_int(value(argument), argument);
        } else if (argument == "--inputs") {
            options.inputs = parse_ids(value(argument), argument);
        } else if (argument == "--targets") {
            options.targets = parse_ids(value(argument), argument);
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argc > 0 ? argv[0] : "ninfer_qwen4_verifier_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }

    if (!weights_set || options.weights.empty()) {
        throw std::invalid_argument("--weights is required; artifacts are never auto-discovered");
    }
    if (options.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (options.warmup <= 0) {
        throw std::invalid_argument("--warmup must be positive so measured tokens are warm");
    }
    if (options.repetitions <= 0) {
        throw std::invalid_argument("--repetitions must be positive");
    }
    if (options.profile && options.repetitions != 1) {
        throw std::invalid_argument("--profile requires --repetitions 1");
    }
    if (options.inputs.size() != options.targets.size()) {
        throw std::invalid_argument("--inputs and --targets must have equal lengths");
    }
    if (options.inputs.size() > static_cast<std::size_t>(verifier::kQsaCapacity)) {
        throw std::invalid_argument("the measured sequence exceeds the 4096-token verifier ceiling");
    }
    const auto valid_id = [](std::int32_t value) { return value >= 0 && value < kVocabulary; };
    if (!std::all_of(options.inputs.begin(), options.inputs.end(), valid_id) ||
        !std::all_of(options.targets.begin(), options.targets.end(), valid_id)) {
        throw std::invalid_argument("every input and target must be in [0,248320)");
    }
    return options;
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double run_sequence(verifier::Program& program, const Options& options,
                    std::vector<std::vector<double>>* token_samples) {
    const Clock::time_point sequence_begin = Clock::now();
    for (std::size_t token = 0; token < options.inputs.size(); ++token) {
        const Clock::time_point token_begin = Clock::now();
        const verifier::TokenResultView result =
            program.execute_token(options.inputs[token], options.targets[token]);
        const Clock::time_point token_end = Clock::now();
        if (!result.gr.empty()) {
            throw std::runtime_error("benchmark Program unexpectedly produced GR snapshots");
        }
        if (result.token_index != static_cast<std::int32_t>(token) ||
            program.frontier() != static_cast<std::int32_t>(token + 1)) {
            throw std::runtime_error("verifier frontier changed during benchmark execution");
        }
        if (token_samples != nullptr) {
            (*token_samples)[token].push_back(elapsed_ms(token_begin, token_end));
        }
    }
    return elapsed_ms(sequence_begin, Clock::now());
}

struct Distribution {
    double mean = 0.0;
    double minimum = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
};

Distribution summarize(std::vector<double> values) {
    if (values.empty()) { throw std::logic_error("cannot summarize an empty measurement"); }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    const std::size_t median_index = values.size() / 2;
    const double median = values.size() % 2 == 0
                              ? 0.5 * (values[median_index - 1] + values[median_index])
                              : values[median_index];
    const std::size_t p95_index = (95 * values.size() + 99) / 100 - 1;
    return {
        .mean = mean,
        .minimum = values.front(),
        .median = median,
        .p95 = values[p95_index],
        .maximum = values.back(),
    };
}

int run(const Options& options) {
    if (!std::filesystem::is_regular_file(options.weights)) {
        throw std::invalid_argument("--weights is not a regular file: " +
                                    options.weights.string());
    }

    const Clock::time_point device_begin = Clock::now();
    ninfer::DeviceContext device(options.device);
    const Clock::time_point device_end = Clock::now();
    const Clock::time_point load_begin = Clock::now();
    std::unique_ptr<verifier::LoadedModel> model =
        verifier::LoadedModel::load(options.weights, device);
    const Clock::time_point load_end = Clock::now();

    const Clock::time_point setup_begin = Clock::now();
    verifier::Program program(*model, device, verifier::DiagnosticSnapshots::Disabled);
    const Clock::time_point setup_end = Clock::now();

    double warmup_reset_ms = 0.0;
    double warmup_token_ms = 0.0;
    {
        ninfer::NvtxRange range("qwen4.verifier.warmup");
        for (int repetition = 0; repetition < options.warmup; ++repetition) {
            const Clock::time_point reset_begin = Clock::now();
            program.reset();
            warmup_reset_ms += elapsed_ms(reset_begin, Clock::now());
            warmup_token_ms += run_sequence(program, options, nullptr);
        }
    }

    std::vector<double> reset_samples;
    std::vector<double> sequence_samples;
    std::vector<double> all_token_samples;
    std::vector<std::vector<double>> token_samples(options.inputs.size());
    reset_samples.reserve(static_cast<std::size_t>(options.repetitions));
    sequence_samples.reserve(static_cast<std::size_t>(options.repetitions));
    all_token_samples.reserve(static_cast<std::size_t>(options.repetitions) * options.inputs.size());

    for (int repetition = 0; repetition < options.repetitions; ++repetition) {
        const Clock::time_point reset_begin = Clock::now();
        program.reset();
        reset_samples.push_back(elapsed_ms(reset_begin, Clock::now()));

        if (options.profile) { CUDA_CHECK(cudaProfilerStart()); }
        {
            ninfer::NvtxRange range("qwen4.verifier.measured_sequence");
            sequence_samples.push_back(run_sequence(program, options, &token_samples));
        }
        if (options.profile) { CUDA_CHECK(cudaProfilerStop()); }
    }
    for (const auto& position : token_samples) {
        all_token_samples.insert(all_token_samples.end(), position.begin(), position.end());
    }

    const Distribution reset = summarize(reset_samples);
    const Distribution sequence = summarize(sequence_samples);
    const Distribution token = summarize(all_token_samples);
    const auto& stats = model->backing().stats();
    const double tokens_per_second = 1000.0 * static_cast<double>(options.inputs.size()) /
                                     sequence.mean;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "format,qwen4_verifier_profile_v1\n";
    std::cout << "artifact," << options.weights.string() << '\n';
    std::cout << "device," << device.props.name << '\n';
    std::cout << "qsa_kv,nvfp4-g16\n";
    std::cout << "diagnostic_snapshots,false\n";
    std::cout << "device_payload_bytes," << stats.h2d_bytes << '\n';
    std::cout << "mapped_tensor_bytes," << stats.mapped_tensor_bytes << '\n';
    std::cout << "sequence_tokens," << options.inputs.size() << '\n';
    std::cout << "warmup_repetitions," << options.warmup << '\n';
    std::cout << "measured_repetitions," << options.repetitions << '\n';
    std::cout << "profile_capture," << (options.profile ? "true" : "false") << '\n';
    std::cout << "device_init_ms," << elapsed_ms(device_begin, device_end) << '\n';
    std::cout << "load_ms," << elapsed_ms(load_begin, load_end) << '\n';
    std::cout << "program_setup_ms," << elapsed_ms(setup_begin, setup_end) << '\n';
    std::cout << "warmup_reset_ms," << warmup_reset_ms << '\n';
    std::cout << "warmup_token_ms," << warmup_token_ms << '\n';
    std::cout << "measured_reset_mean_ms," << reset.mean << '\n';
    std::cout << "sequence_mean_ms," << sequence.mean << '\n';
    std::cout << "sequence_min_ms," << sequence.minimum << '\n';
    std::cout << "sequence_median_ms," << sequence.median << '\n';
    std::cout << "sequence_p95_ms," << sequence.p95 << '\n';
    std::cout << "sequence_max_ms," << sequence.maximum << '\n';
    std::cout << "token_mean_ms," << token.mean << '\n';
    std::cout << "token_min_ms," << token.minimum << '\n';
    std::cout << "token_median_ms," << token.median << '\n';
    std::cout << "token_p95_ms," << token.p95 << '\n';
    std::cout << "token_max_ms," << token.maximum << '\n';
    std::cout << "tokens_per_second," << tokens_per_second << '\n';
    for (std::size_t position = 0; position < token_samples.size(); ++position) {
        const Distribution sample = summarize(token_samples[position]);
        std::cout << "position," << position << ",input," << options.inputs[position]
                  << ",target," << options.targets[position] << ",mean_ms," << sample.mean
                  << ",min_ms," << sample.minimum << ",median_ms," << sample.median
                  << ",p95_ms," << sample.p95 << ",max_ms," << sample.maximum << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "qwen4 verifier benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
