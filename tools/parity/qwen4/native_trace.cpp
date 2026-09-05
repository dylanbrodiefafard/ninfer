#include "core/device.h"
#include "targets/qwen4/verifier.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;

namespace {

constexpr std::int32_t kVocabulary = 248320;
constexpr std::int32_t kExperts = 512;
constexpr std::int32_t kQsaTokenBudget = 2048;
constexpr std::array<std::uint64_t, 3> kNgramMultiplier = {
    23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
constexpr std::array<std::int32_t, 16> kNgramPrime = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr std::array<std::int32_t, 16> kNgramOffset = {
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
    300001275};

struct Options {
    std::filesystem::path weights;
    std::filesystem::path output;
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> probes{0, 7, 10, 60, 221, 227, 438, 521};
    bool probes_explicit = false;
    int device = 0;
};

void usage(const char* executable) {
    std::cout << "usage: " << executable
              << " --weights <qwen4-verifier.ninfer> --tokens <id,id,...>"
                 " --output <trace.txt> [--probes <position,...>] [--device <id>]\n\n"
                 "Executes the complete token sequence with diagnostic snapshots enabled and"
                 " writes post-attention, post-FFN, final-hidden, routing, and NLL records only"
                 " for the requested zero-based input positions. The target for position p is"
                 " tokens[p+1], so at least two token IDs are required.\n";
}

std::vector<std::int32_t> parse_ints(std::string_view text, std::string_view option) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        if (begin == end) {
            throw std::invalid_argument(std::string(option) + " contains an empty integer");
        }
        std::int32_t value = 0;
        const char* first = text.data() + begin;
        const char* last = text.data() + end;
        const auto [parsed, error] = std::from_chars(first, last, value);
        if (error != std::errc{} || parsed != last) {
            throw std::invalid_argument(std::string(option) + " contains a non-integer");
        }
        result.push_back(value);
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    if (result.empty()) { throw std::invalid_argument(std::string(option) + " is empty"); }
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
        } else if (argument == "--tokens") {
            options.tokens = parse_ints(value(argument), argument);
        } else if (argument == "--output") {
            options.output = std::string(value(argument));
        } else if (argument == "--probes") {
            options.probes = parse_ints(value(argument), argument);
            options.probes_explicit = true;
        } else if (argument == "--device") {
            options.device = parse_int(value(argument), argument);
        } else if (argument == "-h" || argument == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_qwen4_native_trace");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.weights.empty() || options.output.empty() || options.tokens.size() < 2) {
        throw std::invalid_argument("--weights, --output, and at least two --tokens IDs are required");
    }
    if (options.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (options.tokens.size() - 1 > static_cast<std::size_t>(verifier::kQsaCapacity)) {
        throw std::invalid_argument("token sequence exceeds the Qwen4 verifier capacity");
    }
    if (!std::all_of(options.tokens.begin(), options.tokens.end(), [](std::int32_t token) {
            return token >= 0 && token < kVocabulary;
        })) {
        throw std::invalid_argument("token IDs must be in [0,248320)");
    }
    std::sort(options.probes.begin(), options.probes.end());
    options.probes.erase(std::unique(options.probes.begin(), options.probes.end()),
                         options.probes.end());
    if (!options.probes_explicit) {
        const auto scored = static_cast<std::int32_t>(options.tokens.size() - 1);
        std::erase_if(options.probes, [scored](std::int32_t probe) { return probe >= scored; });
    }
    if (options.probes.empty() || options.probes.front() < 0 ||
        options.probes.back() >= static_cast<std::int32_t>(options.tokens.size() - 1)) {
        throw std::invalid_argument("probe positions must identify scored input tokens");
    }
    return options;
}

float bf16_to_f32(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::int64_t signed_u64(std::uint64_t value) { return std::bit_cast<std::int64_t>(value); }

std::array<std::int32_t, 16> expected_ple_rows(
    std::int32_t current, const std::array<std::int32_t, 2>& history) {
    const std::int32_t lag1 = history[1];
    const std::int32_t lag2 = lag1 == verifier::kPleResetToken
                                  ? verifier::kPleResetToken
                                  : history[0];
    const std::uint64_t mixed2 =
        static_cast<std::uint64_t>(current) * kNgramMultiplier[0] ^
        static_cast<std::uint64_t>(lag1) * kNgramMultiplier[1];
    const std::uint64_t mixed3 =
        mixed2 ^ static_cast<std::uint64_t>(lag2) * kNgramMultiplier[2];
    std::array<std::int32_t, 16> rows{};
    for (std::size_t head = 0; head < rows.size(); ++head) {
        const std::int64_t mixed = signed_u64(head < 8 ? mixed2 : mixed3);
        std::int64_t remainder = mixed % kNgramPrime[head];
        if (remainder < 0) { remainder += kNgramPrime[head]; }
        rows[head] = kNgramOffset[head] + static_cast<std::int32_t>(remainder);
    }
    return rows;
}

template <class T>
std::vector<T> copy_tensor(const ninfer::Tensor& tensor) {
    std::vector<T> values(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(cudaMemcpy(values.data(), tensor.data, values.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return values;
}

struct Summary {
    double sum = 0.0;
    double sumsq = 0.0;
    double max_abs = 0.0;
    std::uint64_t raw_fnv1a64 = 14695981039346656037ULL;
};

Summary summarize_bf16(const ninfer::Tensor& tensor) {
    const std::vector<std::uint16_t> bits = copy_tensor<std::uint16_t>(tensor);
    Summary summary;
    for (std::uint16_t encoded : bits) {
        const double value = bf16_to_f32(encoded);
        if (!std::isfinite(value)) {
            throw std::runtime_error("native trace encountered non-finite BF16 data");
        }
        summary.sum += value;
        summary.sumsq += value * value;
        summary.max_abs = std::max(summary.max_abs, std::abs(value));
        summary.raw_fnv1a64 ^= encoded & 0xffU;
        summary.raw_fnv1a64 *= 1099511628211ULL;
        summary.raw_fnv1a64 ^= encoded >> 8U;
        summary.raw_fnv1a64 *= 1099511628211ULL;
    }
    return summary;
}

void write_summary(std::ostream& output, std::string_view seam, std::size_t layer,
                   std::int32_t position, const ninfer::Tensor& tensor) {
    const Summary summary = summarize_bf16(tensor);
    output << "TENSOR position=" << position << " seam=" << seam << " layer=" << layer
           << " elements=" << tensor.numel() << " dtype=bf16 sum=" << summary.sum
           << " sumsq=" << summary.sumsq << " max_abs=" << summary.max_abs
           << " raw_fnv1a64=" << std::hex << std::setw(16) << std::setfill('0')
           << summary.raw_fnv1a64 << std::dec << std::setfill(' ') << '\n';
}

void capture_probe(std::ostream& output, std::int32_t position,
                   std::int32_t input_id, std::int32_t target_id,
                   const std::array<std::int32_t, 16>& expected_rows,
                   const verifier::TokenResultView& result) {
    if (result.gr.size() != verifier::kLayerCount ||
        result.routers.size() != verifier::kLayerCount ||
        result.qsa.size() != verifier::kQsaLayerCount) {
        throw std::runtime_error("native trace diagnostic extent changed");
    }
    for (const auto& layer : result.gr) {
        write_summary(output, "attn_residual", layer.layer, position,
                      layer.attention_residual);
        write_summary(output, "ffn_residual", layer.layer, position, layer.ffn_residual);
    }
    write_summary(output, "final_gr", verifier::kLayerCount, position,
                  result.final_hidden);
    const auto ple_row_ids = copy_tensor<std::int32_t>(result.ple_row_ids);
    if (!std::equal(ple_row_ids.begin(), ple_row_ids.end(), expected_rows.begin(),
                    expected_rows.end())) {
        throw std::runtime_error("native trace PLE row IDs disagree with independent n-gram math");
    }
    output << "PLE position=" << position << " input_id=" << input_id
           << " target_id=" << target_id << " row_ids=";
    for (std::size_t head = 0; head < ple_row_ids.size(); ++head) {
        if (head != 0) { output << ','; }
        output << ple_row_ids[head];
    }
    output << '\n';
    for (const auto& qsa : result.qsa) {
        const auto selected_count = copy_tensor<std::int32_t>(qsa.selected_count).front();
        const std::int32_t visible_count = position + 1;
        const std::int32_t expected_count =
            visible_count <= kQsaTokenBudget
                ? visible_count
                : kQsaTokenBudget + visible_count % 4;
        if (selected_count != expected_count || selected_count < 0 ||
            selected_count > static_cast<std::int32_t>(qsa.selected_ids.numel())) {
            throw std::runtime_error("native trace encountered an invalid QSA selection count");
        }
        const auto ids = copy_tensor<std::int32_t>(qsa.selected_ids);
        std::vector<bool> seen(static_cast<std::size_t>(visible_count));
        for (std::int32_t rank = 0; rank < selected_count; ++rank) {
            const std::int32_t id = ids[static_cast<std::size_t>(rank)];
            if (id < 0 || id >= visible_count || seen[static_cast<std::size_t>(id)]) {
                throw std::runtime_error("native trace encountered invalid or duplicate QSA IDs");
            }
            seen[static_cast<std::size_t>(id)] = true;
        }
        if (visible_count <= kQsaTokenBudget &&
            !std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })) {
            throw std::runtime_error("native trace QSA selection omitted a visible token");
        }
        if (!std::all_of(ids.begin() + selected_count, ids.end(),
                         [](std::int32_t id) { return id == -1; })) {
            throw std::runtime_error("native trace QSA selection padding is not -1");
        }
        output << "QSA position=" << position << " layer=" << qsa.layer
               << " selected_count=" << selected_count << " ids=";
        for (std::int32_t rank = 0; rank < selected_count; ++rank) {
            if (rank != 0) { output << ','; }
            output << ids[static_cast<std::size_t>(rank)];
        }
        output << '\n';
    }
    for (const auto& router : result.routers) {
        const auto ids = copy_tensor<std::int32_t>(router.selected_ids);
        const auto weights = copy_tensor<float>(router.selected_weights);
        if (ids.size() != weights.size() || ids.empty()) {
            throw std::runtime_error("native trace encountered malformed router diagnostics");
        }
        std::array<bool, kExperts> seen{};
        double weight_sum = 0.0;
        for (std::size_t rank = 0; rank < ids.size(); ++rank) {
            const std::int32_t id = ids[rank];
            if (id < 0 || id >= kExperts || seen[static_cast<std::size_t>(id)]) {
                throw std::runtime_error("native trace encountered invalid or duplicate router IDs");
            }
            seen[static_cast<std::size_t>(id)] = true;
            if (!std::isfinite(weights[rank]) || weights[rank] < 0.0F) {
                throw std::runtime_error("native trace encountered an invalid router weight");
            }
            weight_sum += weights[rank];
        }
        constexpr double kWeightSumTolerance =
            128.0 * std::numeric_limits<float>::epsilon();
        if (std::abs(weight_sum - 1.0) > kWeightSumTolerance) {
            throw std::runtime_error("native trace router weights are not normalized");
        }
        output << "ROUTER position=" << position << " layer=" << router.layer << " ids=";
        for (std::size_t rank = 0; rank < ids.size(); ++rank) {
            if (rank != 0) { output << ','; }
            output << ids[rank];
        }
        output << " weights=";
        for (std::size_t rank = 0; rank < weights.size(); ++rank) {
            if (rank != 0) { output << ','; }
            output << weights[rank];
        }
        output << '\n';
    }
    const float nll = copy_tensor<float>(result.nll).front();
    if (!std::isfinite(nll) || nll < 0.0F) {
        throw std::runtime_error("native trace encountered an invalid NLL");
    }
    output << "NLL position=" << position << " input_id=" << input_id
           << " target_id=" << target_id << " value=" << nll << '\n';
}

int run(const Options& options) {
    if (!std::filesystem::is_regular_file(options.weights)) {
        throw std::invalid_argument("--weights is not a regular file");
    }
    std::ofstream output(options.output, std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot open --output path"); }
    output << std::setprecision(17);
    output << "format,ninfer.qwen4.native_boundary_trace.v1\n"
           << "qsa_kv,nvfp4-g16\n"
           << "token_count," << options.tokens.size() << '\n'
           << "scored_count," << options.tokens.size() - 1 << '\n';

    ninfer::DeviceContext device(options.device);
    const std::unique_ptr<verifier::LoadedModel> model =
        verifier::LoadedModel::load(options.weights, device);
    verifier::Program program(*model, device, verifier::DiagnosticSnapshots::Enabled);
    program.reset();
    std::array<std::int32_t, 2> ple_history{
        verifier::kPleResetToken, verifier::kPleResetToken};
    std::size_t probe = 0;
    for (std::size_t position = 0; position + 1 < options.tokens.size(); ++position) {
        const auto expected_rows = expected_ple_rows(options.tokens[position], ple_history);
        const verifier::TokenResultView result =
            program.execute_token(options.tokens[position], options.tokens[position + 1]);
        if (probe < options.probes.size() &&
            options.probes[probe] == static_cast<std::int32_t>(position)) {
            capture_probe(output, static_cast<std::int32_t>(position),
                          options.tokens[position], options.tokens[position + 1], expected_rows,
                          result);
            ++probe;
        }
        ple_history = {ple_history[1], options.tokens[position]};
    }
    if (probe != options.probes.size()) {
        throw std::runtime_error("not every requested probe was captured");
    }
    if (!output) { throw std::runtime_error("failed writing native boundary trace"); }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Qwen4 native trace failed: " << error.what() << '\n';
        return 1;
    }
}
