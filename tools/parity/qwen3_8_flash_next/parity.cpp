// Qwen3.8-Flash-Next teacher-forced parity driver (milestone M1 of
// docs/maintainer/qwen3.8-flash-next-plan.md). It runs the target's Forward on token sequences and
// writes raw dumps for comparison against the study harness (fnstudy/forward.py, Runner with
// last_logits / keep_residual), checks chunked prefill against single-token steps, or
// teacher-forces a token panel through the decode path (decode-forced mode).

#include "npy.h"
#include "ninfer/engine.h"
#include "product/speculative_options.h"

#include "load/binder.h"
#include "artifact/reader.h"
#include "runtime/forward.h"
#include "runtime/program.h"

#include "core/device.h"
#include "core/expert_helper_pci.h"
#include "targets/startup_options.h"
#include "ninfer/ops/moe_experts.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <thread>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fn  = ninfer::targets::qwen3_8_flash_next;
namespace fnt = ninfer::tools::flash_next;

constexpr std::string_view kUsage = R"(usage: ninfer_qwen3_8_flash_next_parity [options]

Required:
  --artifact PATH      qwen3.8-flash-next .ninfer artifact
  --ngram-table PATH   int8_row n-gram table (ngram_table.int8.bin)
  --expert-helper-pci DOMAIN:BUS:DEVICE.FUNCTION
                      optional gfx1201 decode helper; requires HIP-enabled build
                      experimental: helper inference qualification remains pending
  --tokens PATH        token ids: a C-order int32 .npy [S,T], or raw little-endian I32 with
                       --sequences S sequences of equal length T
  --out DIR            output directory (created)

Options:
  --sequences S        sequences in a raw token file (default 1)
  --mode MODE          prefill (default), split, decode-forced, bench, trace, lanes-check,
                       lanes-bench, lanes-trace or dflash-acceptance
  --logits WHICH       last (default) or all (prefill mode)
  --residuals          also dump the residual after every layer (prefill mode)
  --split K            split mode: prefill T-K tokens, then K single-token steps (default 8)
  --chunk N            widest prefill sub-chunk (default 512)
  --super-chunk N      widest layer-major prefill super-chunk, at least --chunk (default --chunk)
  --context N          QSA state capacity in tokens (default T; bench and trace modes: at least
                       the prompt and every step)
  --budget BLOCKS      QSA selection budget: 512 (default), 1024 or 2048 blocks
  --prompt-lengths L,..  bench mode: prompt lengths (default 1024,4096,16384), taken from the
                       token file's sequences concatenated in order
  --decode N           bench and trace modes: greedy decode steps after each prompt (default 2048)
  --bucket N           trace mode: decode steps per expert-cache counter bucket (default 128)
  --teacher            trace mode: step through the token file's continuation of the prompt
                       instead of greedy tokens (teacher-forced routes)
  --profile-decode     bench mode: bracket the timed decode steps with cudaProfilerStart/Stop
                       (nsys --capture-range=cudaProfilerApi)
  --expert-cache-mib N expert-cache slot MiB (default: all free device memory after the rest)
  --window F           each of the two prefill streaming windows holds fraction F in [0,1] of the
                       largest layer's expert records (default 1; 0 disables streaming)
  --window-min-tokens N  narrowest super-chunk that streams through the windows (default 512)
  --expert-staging on|off  predictive expert staging of decode and verification rounds (default
                       on); it changes when records cross PCIe, never a result
  --calib-counts PATH  CUDA-only cache warm-start '<i8' .npy [48,512] counts
                      helper mode uses its compiled profile; trace modes are CUDA-only
  --kv-format F        QSA state format: nvfp4 (default) or bf16
  --lanes L,..         lanes-check: batch sizes (default 2,4,12); lanes-bench: concurrencies
  --kv-capacity N      lanes-bench: shared QSA pool tokens (default: every lane's context)
  --spec none|dflash   lanes-bench/lanes-trace: disabled (default) or native DFlash2
  --draft-tokens K     fixed K in [0,3]; lanes-bench also uses it as the adaptive ceiling
  --adaptive-draft     lanes-bench: batch-size K table with request-local step-down
  --interleave N       lanes-check: interleave an N-token prefill with the B=2 decode rounds
  --interleave-tokens PATH  raw I32 tokens of the interleaved prompt (N + 1024 tokens)
  --prompt-offset K    lanes-bench: lane i takes prompt i + K (default 0)

Prefill mode prefills every sequence independently from an empty state and writes:
  logits_s<s>.bf16           --logits last: BF16 [1,248320], the last position's logits
  logprobs.npy               --logits all: float16 [S,T-1,248320], the log-softmax over all
                             248320 head rows at positions 0..T-2 (each predicting the next
                             token), the study harness's parity-reference layout
  residual_s<s>_l<L>.bf16    BF16 [T,4,2560] after layer L (--residuals), token, stream, feature
Split mode compares the logits of positions T-K..T-1 from one teacher-forced prefill (chunked
by --chunk) with those of a prefill of T-K tokens followed by K single-token steps, and writes
split_s<s>.json: per position the max and mean absolute logit difference, KL(prefill || steps)
and top-1 agreement.
Decode-forced mode writes the prefill mode's logprobs.npy (float16 [S,T-1,248320], identical
layout and log-softmax) but produces it through the decode path the Engine uses, so an expert
helper participates: per sequence, position 0 comes from a one-token prefill (always local on
the primary), and positions 1..T-2 from T-2 B=1 graph-replayed decode steps whose input token is
forced to the file's token instead of the sampled one. It needs T >= 2, --logits is ignored, and
it writes decode_forced.json (helper, sequences, length).
Bench mode times, per prompt length, the prefill (tok/s) and greedy decode through the B=1 decode
graph, and reports for two windows, the first 256 timed steps and all of them (steady state):
decode tok/s, per token the graph replay, host n-gram gather and remaining host time, and the
expert-cache hit rate, misses per codec and PCIe record bytes that the device counters recorded
over the window. It checks for the first prompt length that eager decode
rounds reproduce the graph's tokens (the count of agreeing steps; -2 when the warm-up token
differs, -1 for other lengths), and writes bench.json. One untimed decode step precedes the timed
ones.
Lanes-check mode compares lane-batched decode with solo lanes (see run_lanes_check) and writes
lanes_check.json; lanes-bench uses the public Engine and measures the monotonic committed-token
counter over the wall interval after all initial prefills, including natural request drain.
It writes lanes_bench.json and row-major generated tokens; its startup concurrency equals C.
Private cache/window and profiler controls are not supported there. Lanes-trace records expert routes
and exact post-prefill cache/frequency state for policy replay (see run_lanes_trace), including every
verification column with --spec dflash --draft-tokens K; it requires fixed K. Both modes take the
prompt length from the first --prompt-lengths value. --decode counts outputs per request for
lanes-bench and rounds for lanes-trace.
Dflash-acceptance mode uses every [S,2048] input sequence at frontiers 64,128,...,1984, with
K=3, temperature 1, top-k 20, top-p .95 and verifier-selected anchors. It writes realized native
conditional accept-walk counts to dflash_acceptance.json, not integrated acceptance probabilities.
Trace mode prefills the first --prompt-lengths prompt, saves the expert cache's post-prefill
state, then runs --decode eager single-token steps (the decode graph's kernels and cache
transitions) on greedy tokens, or with --teacher on the tokens that follow the prompt, and writes:
  routes.npy                 int32 [N,48,10]: the routed expert ids of every step and layer
  codecs.npy                 uint8 [48,512]: each expert's record codec (0 NVFP4, 1 F3)
  owner_p<c>.npy             int32 [slots]: post-prefill owner (l * 512 + e, or -1) of each slot
                             of pool c (0 NVFP4, 1 F3)
  referenced_p<c>.npy        uint8 [slots]: post-prefill CLOCK reference bits of pool c
  clock_hands.npy            int32 [2]: post-prefill CLOCK hands
  trace.json                 the slot counts and, per bucket of --bucket steps, the device
                             counters' lookups and misses per codec
)";

struct Options {
    std::filesystem::path artifact;
    std::filesystem::path ngram_table;
    std::optional<std::string> expert_helper_pci;
    std::filesystem::path tokens;
    std::filesystem::path out;
    std::int64_t sequences      = 1;
    bool split_mode             = false;
    bool bench_mode             = false;
    bool trace_mode             = false;
    bool decode_forced_mode     = false;
    bool lanes_check_mode       = false;
    bool lanes_bench_mode       = false;
    bool dflash_acceptance_mode = false;
    ninfer::SpeculativeOptions speculative{};
    bool lanes_trace_mode = false;
    std::int32_t bucket   = 128;
    bool teacher          = false;
    std::vector<std::int32_t> prompt_lengths{1024, 4096, 16384};
    std::int32_t decode_tokens = 2048;
    bool profile_decode        = false;
    bool all_logits            = false;
    bool residuals             = false;
    std::int32_t split         = 8;
    std::int32_t chunk         = 512;
    std::int32_t super_chunk   = 0; // 0: --chunk
    std::int32_t context       = 0;
    std::int32_t budget        = 512;
    // Expert-cache MiB; negative sizes the cache from free device memory.
    std::int64_t expert_cache_mib    = -1;
    double window                    = 1.0;
    std::int32_t window_min_tokens   = 512;
    fn::ExpertStaging expert_staging = fn::ExpertStaging::Enabled;
    std::filesystem::path calib_counts;
    ninfer::ops::QsaKvFormat kv_format = ninfer::ops::QsaKvFormat::NVFP4G16;
    // Lane modes: the batch sizes (lanes-check) or concurrencies (lanes-bench).
    std::vector<std::int32_t> lane_counts{2, 4, 12};
    std::optional<std::int64_t> kv_capacity;
    // lanes-check: the prompt whose super-chunks interleave with the decode rounds (0: none),
    // taken from --interleave-tokens.
    std::int32_t interleave_tokens = 0;
    std::filesystem::path interleave_file;
    // lanes-bench: lane i's prompt is prompt index i + prompt_offset of the token stream.
    std::int32_t prompt_offset = 0;
};

std::int32_t parse_int(std::string_view text, std::string_view flag) {
    const std::string value(text);
    const auto invalid = [&] {
        return std::invalid_argument(std::string(flag) + ": invalid value '" + value + "'");
    };
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw invalid();
    }
    try {
        const long long parsed = std::stoll(value);
        if (parsed > INT32_MAX) { throw invalid(); }
        return static_cast<std::int32_t>(parsed);
    } catch (const std::out_of_range&) { throw invalid(); }
}

std::vector<std::int32_t> parse_list(std::string_view text, std::string_view flag) {
    std::vector<std::int32_t> values;
    const std::string list(text);
    for (std::size_t at = 0; at <= list.size();) {
        const std::size_t comma = std::min(list.find(',', at), list.size());
        values.push_back(parse_int(list.substr(at, comma - at), flag));
        at = comma + 1;
    }
    return values;
}

Options parse(std::span<char*> args) {
    Options options;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view flag = args[i];
        const auto value            = [&]() -> std::string_view {
            if (i + 1 >= args.size()) {
                throw std::invalid_argument(std::string(flag) + " needs a value");
            }
            return args[++i];
        };
        if (flag == "--artifact") {
            options.artifact = value();
        } else if (flag == "--ngram-table") {
            options.ngram_table = value();
        } else if (flag == "--tokens") {
            options.tokens = value();
        } else if (flag == "--out") {
            options.out = value();
        } else if (flag == "--sequences") {
            options.sequences = parse_int(value(), flag);
        } else if (flag == "--mode") {
            const std::string_view mode = value();
            if (mode != "prefill" && mode != "split" && mode != "decode-forced" &&
                mode != "bench" && mode != "trace" && mode != "lanes-check" &&
                mode != "lanes-bench" && mode != "lanes-trace" && mode != "dflash-acceptance") {
                throw std::invalid_argument("--mode: prefill, split, bench, trace, lanes-check, "
                                            "lanes-bench, lanes-trace or dflash-acceptance");
            }
            options.split_mode             = mode == "split";
            options.decode_forced_mode     = mode == "decode-forced";
            options.bench_mode             = mode == "bench";
            options.trace_mode             = mode == "trace";
            options.lanes_check_mode       = mode == "lanes-check";
            options.lanes_bench_mode       = mode == "lanes-bench";
            options.dflash_acceptance_mode = mode == "dflash-acceptance";
            options.lanes_trace_mode       = mode == "lanes-trace";
        } else if (flag == "--logits") {
            const std::string_view which = value();
            if (which != "last" && which != "all") {
                throw std::invalid_argument("--logits: last or all");
            }
            options.all_logits = which == "all";
        } else if (flag == "--residuals") {
            options.residuals = true;
        } else if (flag == "--split") {
            options.split = parse_int(value(), flag);
        } else if (flag == "--super-chunk") {
            options.super_chunk = parse_int(value(), flag);
        } else if (flag == "--chunk") {
            options.chunk = parse_int(value(), flag);
        } else if (flag == "--context") {
            options.context = parse_int(value(), flag);
        } else if (flag == "--budget") {
            options.budget = parse_int(value(), flag);
        } else if (flag == "--prompt-lengths") {
            options.prompt_lengths = parse_list(value(), flag);
        } else if (flag == "--spec") {
            options.speculative.backend = ninfer::product::parse_speculative_backend(value());
        } else if (flag == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(parse_int(value(), flag));
        } else if (flag == "--adaptive-draft") {
            options.speculative.adaptive_draft = true;
        } else if (flag == "--expert-helper-pci") {
            options.expert_helper_pci = ninfer::normalize_helper_pci(value());
        } else if (flag == "--profile-decode") {
            options.profile_decode = true;
        } else if (flag == "--decode") {
            options.decode_tokens = parse_int(value(), flag);
        } else if (flag == "--teacher") {
            options.teacher = true;
        } else if (flag == "--bucket") {
            options.bucket = parse_int(value(), flag);
        } else if (flag == "--window") {
            const std::string text(value());
            std::size_t used = 0;
            try {
                options.window = std::stod(text, &used);
            } catch (const std::exception&) { used = 0; }
            if (used != text.size() || !(options.window >= 0.0 && options.window <= 1.0)) {
                throw std::invalid_argument("--window: a fraction in [0,1]");
            }
        } else if (flag == "--window-min-tokens") {
            options.window_min_tokens = parse_int(value(), flag);
        } else if (flag == "--expert-staging") {
            const std::string_view staging = value();
            if (staging != "on" && staging != "off") {
                throw std::invalid_argument("--expert-staging: on or off");
            }
            options.expert_staging =
                staging == "on" ? fn::ExpertStaging::Enabled : fn::ExpertStaging::Disabled;
        } else if (flag == "--expert-cache-mib") {
            options.expert_cache_mib = parse_int(value(), flag);
        } else if (flag == "--calib-counts") {
            options.calib_counts = value();
        } else if (flag == "--kv-format") {
            const std::string_view format = value();
            if (format != "nvfp4" && format != "bf16") {
                throw std::invalid_argument("--kv-format: nvfp4 or bf16");
            }
            options.kv_format = format == "bf16" ? ninfer::ops::QsaKvFormat::BF16
                                                 : ninfer::ops::QsaKvFormat::NVFP4G16;
        } else if (flag == "--lanes") {
            options.lane_counts = parse_list(value(), flag);
        } else if (flag == "--kv-capacity") {
            options.kv_capacity = parse_int(value(), flag);
        } else if (flag == "--interleave") {
            options.interleave_tokens = parse_int(value(), flag);
        } else if (flag == "--interleave-tokens") {
            options.interleave_file = value();
        } else if (flag == "--prompt-offset") {
            options.prompt_offset = parse_int(value(), flag);
        } else {
            throw std::invalid_argument("unknown argument " + std::string(flag));
        }
    }
    if (options.artifact.empty() || options.ngram_table.empty() || options.tokens.empty() ||
        options.out.empty()) {
        throw std::invalid_argument("--artifact, --ngram-table, --tokens and --out are required");
    }
    ninfer::targets::validate_expert_helper_option(options.expert_helper_pci, "qwen3.8-flash-next");
    if (options.expert_helper_pci &&
        (!options.calib_counts.empty() || options.trace_mode || options.lanes_trace_mode)) {
        throw std::invalid_argument(
            "--expert-helper-pci uses the compiled placement profile; --calib-counts and "
            "cache trace modes are CUDA-only");
    }
    if (options.sequences < 1) { throw std::invalid_argument("--sequences must be positive"); }
    if (options.decode_tokens < 1) { throw std::invalid_argument("--decode must be positive"); }
    if (options.bucket < 1) { throw std::invalid_argument("--bucket must be positive"); }
    fn::validate_speculative_options(options.speculative);
    if (!options.lanes_bench_mode && !options.lanes_trace_mode &&
        options.speculative.backend != ninfer::SpeculativeBackend::None) {
        throw std::invalid_argument(
            "--spec applies only to lanes-bench and lanes-trace; dflash-acceptance fixes K=3");
    }
    return options;
}

ninfer::ops::QsaSelectBudget select_budget(std::int32_t blocks) {
    switch (blocks) {
    case 512:
        return ninfer::ops::QsaSelectBudget::Blocks512;
    case 1024:
        return ninfer::ops::QsaSelectBudget::Blocks1024;
    case 2048:
        return ninfer::ops::QsaSelectBudget::Blocks2048;
    default:
        throw std::invalid_argument("--budget must be 512, 1024 or 2048");
    }
}

struct TokenFile {
    std::vector<std::int32_t> tokens;
    std::int64_t sequences = 1;
};

// Reads a C-order '<i4' .npy [S,T], or raw little-endian I32 ids holding `raw_sequences`
// sequences.
TokenFile read_tokens(const std::filesystem::path& path, std::int64_t raw_sequences) {
    TokenFile out{.tokens = {}, .sequences = raw_sequences};
    std::vector<std::byte> bytes;
    if (path.extension() == ".npy") {
        fnt::NpyArray array = fnt::read_npy(path, "<i4", sizeof(std::int32_t));
        if (array.shape.size() != 2 || array.shape[0] < 1 || array.shape[1] < 1) {
            throw std::runtime_error(path.string() + " must hold a nonempty [S,T] array");
        }
        out.sequences = array.shape[0];
        bytes         = std::move(array.data);
    } else {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) { throw std::runtime_error("cannot open " + path.string()); }
        bytes.resize(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (!file) { throw std::runtime_error("cannot read " + path.string()); }
    }
    if (bytes.empty() || bytes.size() % sizeof(std::int32_t) != 0) {
        throw std::runtime_error(path.string() + " is not a nonempty I32 array");
    }
    out.tokens.resize(bytes.size() / sizeof(std::int32_t));
    std::memcpy(out.tokens.data(), bytes.data(), bytes.size());
    return out;
}

// Calibration routing counts: a C-order '<i8' .npy [48,512] of nonnegative counts.
std::vector<std::uint64_t> read_counts(const std::filesystem::path& path) {
    const fnt::NpyArray array = fnt::read_npy(path, "<i8", sizeof(std::int64_t));
    if (array.shape != std::vector<std::int64_t>{fn::kLayers, ninfer::ops::kMoeExperts}) {
        throw std::runtime_error(path.string() + " must hold [48,512] counts");
    }
    std::vector<std::int64_t> counts(array.data.size() / sizeof(std::int64_t));
    std::memcpy(counts.data(), array.data.data(), array.data.size());
    std::vector<std::uint64_t> out;
    out.reserve(counts.size());
    for (const std::int64_t count : counts) {
        if (count < 0) { throw std::runtime_error(path.string() + " holds a negative count"); }
        out.push_back(static_cast<std::uint64_t>(count));
    }
    return out;
}

// IEEE binary16 bits of x, rounded to nearest even; overflow saturates to infinity.
std::uint16_t half_bits(float x) {
    const auto bits               = std::bit_cast<std::uint32_t>(x);
    const std::uint32_t sign      = (bits >> 16U) & 0x8000U;
    const std::uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude >= 0x7f800000U) { // infinity or NaN
        return static_cast<std::uint16_t>(sign | 0x7c00U | (magnitude > 0x7f800000U ? 0x200U : 0U));
    }
    if (magnitude >= 0x477ff000U) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    if (magnitude < 0x38800000U) { // binary16 subnormal or zero: scale into the 2^-24 grid
        const float scaled = std::bit_cast<float>(magnitude) * 16777216.0F; // exact, < 1024
        return static_cast<std::uint16_t>(sign |
                                          static_cast<std::uint32_t>(std::nearbyint(scaled)));
    }
    const std::uint32_t rebiased = magnitude - 0x38000000U; // exponent bias 127 -> 15
    const std::uint32_t rounding = 0xfffU + ((rebiased >> 13U) & 1U);
    return static_cast<std::uint16_t>(sign | ((rebiased + rounding) >> 13U));
}

// Device tensor contents in storage order, after synchronizing the forward stream.
std::vector<std::uint16_t> download_bf16(const ninfer::Tensor& tensor, cudaStream_t stream) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(
        cudaMemcpyAsync(host.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}

void append(const std::filesystem::path& path, std::span<const std::uint16_t> values) {
    std::ofstream file(path, std::ios::binary | std::ios::app);
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
    if (!file) { throw std::runtime_error("cannot write " + path.string()); }
}

double bf16_value(std::uint16_t bits) {
    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U));
}

// float16 log-softmax of one position's BF16 logits, evaluated in FP64.
std::vector<std::uint16_t> log_softmax_half(std::span<const std::uint16_t> logits) {
    double maximum = -INFINITY;
    for (const std::uint16_t bits : logits) { maximum = std::max(maximum, bf16_value(bits)); }
    double sum = 0.0;
    for (const std::uint16_t bits : logits) { sum += std::exp(bf16_value(bits) - maximum); }
    const double log_sum = std::log(sum);
    std::vector<std::uint16_t> out(logits.size());
    for (std::size_t v = 0; v < logits.size(); ++v) {
        out[v] = half_bits(static_cast<float>(bf16_value(logits[v]) - maximum - log_sum));
    }
    return out;
}

struct Comparison {
    double max_abs  = 0.0;
    double mean_abs = 0.0;
    double kl       = 0.0;
    bool top1_agree = false;
};

// Compares two BF16 logit vectors of one position: reference p (prefill) against q (steps).
Comparison compare(std::span<const std::uint16_t> p, std::span<const std::uint16_t> q) {
    Comparison out;
    double p_max      = -INFINITY;
    double q_max      = -INFINITY;
    std::size_t p_arg = 0;
    std::size_t q_arg = 0;
    for (std::size_t v = 0; v < p.size(); ++v) {
        const double a = bf16_value(p[v]);
        const double b = bf16_value(q[v]);
        const double d = std::abs(a - b);
        out.max_abs    = std::max(out.max_abs, d);
        out.mean_abs += d;
        if (a > p_max) {
            p_max = a;
            p_arg = v;
        }
        if (b > q_max) {
            q_max = b;
            q_arg = v;
        }
    }
    out.mean_abs /= static_cast<double>(p.size());
    out.top1_agree = p_arg == q_arg;
    double p_sum   = 0.0;
    double q_sum   = 0.0;
    for (std::size_t v = 0; v < p.size(); ++v) {
        p_sum += std::exp(bf16_value(p[v]) - p_max);
        q_sum += std::exp(bf16_value(q[v]) - q_max);
    }
    const double p_log_sum = std::log(p_sum);
    const double q_log_sum = std::log(q_sum);
    for (std::size_t v = 0; v < p.size(); ++v) {
        const double log_p = bf16_value(p[v]) - p_max - p_log_sum;
        const double log_q = bf16_value(q[v]) - q_max - q_log_sum;
        out.kl += std::exp(log_p) * (log_p - log_q);
    }
    return out;
}

fn::ProgramOptions program_options_for(const Options& options, std::int32_t context,
                                       fn::LogitColumns logits,
                                       std::span<const std::uint64_t> calib_counts,
                                       std::int32_t lanes                      = 1,
                                       std::optional<std::int64_t> kv_capacity = std::nullopt) {
    fn::ProgramOptions program_options{
        .ngram_table        = options.ngram_table,
        .context_tokens     = context,
        .lanes              = lanes,
        .kv_capacity_tokens = kv_capacity,
        .kv_format          = options.kv_format,
        .forward =
            fn::ForwardOptions{
                .max_chunk_tokens = options.chunk,
                .super_chunk_tokens =
                    options.super_chunk == 0 ? options.chunk : options.super_chunk,
                .select_budget = select_budget(options.budget),
                .logits        = logits,
                .trace_verify_routes =
                    options.lanes_trace_mode &&
                    options.speculative.backend != ninfer::SpeculativeBackend::None,
                .window_min_tokens = options.window_min_tokens,
                .expert_staging    = options.expert_staging,
            },
        .speculative          = options.speculative,
        .expert_window_layers = options.window,
        .expert_cache_bytes   = std::nullopt,
        .calib_counts         = calib_counts,
        .expert_helper_pci    = options.expert_helper_pci,
    };
    if (options.expert_cache_mib >= 0) {
        program_options.expert_cache_bytes = static_cast<std::size_t>(options.expert_cache_mib)
                                             << 20U;
    }
    return program_options;
}

void print_plan(fn::Program& program) {
    const fn::ExpertSlotPlan& slots = program.experts().plan();
    std::cerr << program.lanes() << " lanes, QSA pool " << program.kv_capacity_tokens()
              << " tokens, state " << program.state().device_bytes() << " B, activations "
              << program.forward().activation_bytes() << " B, workspace "
              << program.forward().workspace_bytes() << " B, streaming windows "
              << program.forward().window_bytes() << " B, expert slots " << slots.slots[0]
              << " NVFP4 + " << slots.slots[1] << " F3 (" << slots.bytes() << " B) + "
              << program.experts().flex_slots() << " F3 over idle QSA extents\n";
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Decode rounds of the bench's first reporting window; the second covers every timed round.
constexpr std::int32_t kFirstWindowSteps = 256;

// Timing and device expert-cache counters of a window of timed decode rounds of `rows` lanes.
struct DecodeWindow {
    std::int32_t rows     = 1;
    std::int32_t steps    = 0;
    double seconds        = 0.0; // wall time of the window's rounds
    double replay_seconds = 0.0;
    double gather_seconds = 0.0;
    fn::ExpertCacheCounts counts;

    void add(const fn::RoundTiming& round) {
        ++steps;
        replay_seconds += static_cast<double>(round.replay_ms) * 1e-3;
        gather_seconds += round.gather_seconds;
    }

    // One summary line to `log` and one JSON object to `json`; per-round values are per token at
    // one lane. Replay time spans the round's graphs including any exposed n-gram wait, and the
    // host gather overlaps it, so the host gap is the round time outside the replay interval.
    void report(std::ostream& log, std::ostream& json) const {
        // PCIe time of the misses at the measured mapped-read rate of the plan (26.8 GB/s).
        constexpr double kMappedReadBytesPerSecond = 26.8e9;
        std::int64_t lookups                       = 0;
        std::int64_t misses                        = 0;
        std::int64_t miss_bytes                    = 0;
        for (std::size_t codec = 0; codec < counts.lookups.size(); ++codec) {
            lookups += counts.lookups[codec];
            misses += counts.misses[codec];
            miss_bytes +=
                counts.misses[codec] * ninfer::ops::moe_expert_record_bytes(
                                           static_cast<ninfer::ops::MoeExpertCodec>(codec));
        }
        const double hit_rate =
            lookups == 0 ? 0.0 : 1.0 - static_cast<double>(misses) / static_cast<double>(lookups);
        const double per_round_ms         = seconds * 1e3 / steps;
        const double replay_ms            = replay_seconds * 1e3 / steps;
        const double gather_ms            = gather_seconds * 1e3 / steps;
        const double miss_bytes_per_round = static_cast<double>(miss_bytes) / steps;
        const double aggregate            = static_cast<double>(rows) * steps / seconds;
        const auto misses_per_round       = [&](ninfer::ops::MoeExpertCodec codec) {
            return static_cast<double>(counts.misses[static_cast<std::size_t>(codec)]) / steps;
        };
        log << steps << " rounds of " << rows << ": aggregate " << aggregate << " tok/s, per lane "
            << steps / seconds << " tok/s (per round " << per_round_ms << " ms: replay "
            << replay_ms << ", n-gram gather " << gather_ms << " overlapping replay, host "
            << per_round_ms - replay_ms << "); expert-cache hit rate " << hit_rate << ", PCIe "
            << miss_bytes_per_round / 1e6 << " MB/round\n";
        json << "{\"rounds\": " << steps << ", \"lanes\": " << rows
             << ", \"aggregate_tokens_per_second\": " << aggregate
             << ", \"lane_tokens_per_second\": " << steps / seconds
             << ", \"round_ms\": " << per_round_ms << ", \"replay_ms_per_round\": " << replay_ms
             << ", \"ngram_gather_ms_per_round\": " << gather_ms
             << ", \"host_gap_ms_per_round\": " << per_round_ms - replay_ms
             << ", \"expert_cache_hit_rate\": " << hit_rate << ", \"nvfp4_misses_per_round\": "
             << misses_per_round(ninfer::ops::MoeExpertCodec::Nvfp4G16)
             << ", \"f3_misses_per_round\": "
             << misses_per_round(ninfer::ops::MoeExpertCodec::F3G32)
             << ", \"pcie_record_bytes_per_round\": " << miss_bytes_per_round
             << ", \"pcie_ms_per_round_at_26_8_gb_per_s\": "
             << miss_bytes_per_round / kMappedReadBytesPerSecond * 1e3 << "}";
    }
};

// Prefills `prompt` on an open lane through prefill_step and returns the first sampled token.
std::int32_t prefill_prompt(fn::Program& program, fn::Lane lane,
                            std::span<const std::int32_t> prompt) {
    for (;;) {
        const fn::PrefillStep step = program.prefill_step(lane, prompt);
        prompt                     = prompt.subspan(static_cast<std::size_t>(step.consumed));
        if (step.complete) { return step.next_token; }
    }
}

// Opens a greedy lane reserving `tokens`; throws when none is free.
fn::Lane open_greedy(fn::Program& program, std::int64_t tokens) {
    const std::optional<fn::Lane> lane = program.open_lane(tokens, fn::LaneSampling{});
    if (!lane.has_value()) {
        throw std::runtime_error("no lane or pages for " + std::to_string(tokens) + " tokens");
    }
    return *lane;
}

// Greedy device sampling configurations of `rows` rows (no counts, no mask).
std::vector<ninfer::ops::SamplingConfig> greedy_configs(std::size_t rows) {
    return std::vector<ninfer::ops::SamplingConfig>(rows);
}

// Runs `rounds` timed greedy decode rounds over `lanes` from `tokens` (updated to the last
// sampled ones), filling the first-window and whole-run statistics; returns every round's tokens.
std::vector<std::int32_t> timed_rounds(fn::Program& program, std::span<const fn::Lane> lanes,
                                       std::vector<std::int32_t>& tokens, std::int32_t rounds,
                                       bool profile, std::array<DecodeWindow, 2>& windows) {
    fn::Forward& forward = program.forward();
    const auto rows      = static_cast<std::int32_t>(lanes.size());
    windows              = {DecodeWindow{.rows = rows}, DecodeWindow{.rows = rows}};
    std::vector<std::int32_t> next(lanes.size());
    std::vector<std::int32_t> history;
    history.reserve(static_cast<std::size_t>(rounds) * lanes.size());
    const std::int32_t first_steps = std::min(kFirstWindowSteps, rounds);
    forward.clear_expert_cache_counts();
    if (profile) { CUDA_CHECK(cudaProfilerStart()); }
    const auto start = std::chrono::steady_clock::now();
    for (std::int32_t round = 0; round < rounds; ++round) {
        program.decode(lanes, tokens, next);
        if (round < first_steps) { windows[0].add(program.last_round()); }
        windows[1].add(program.last_round());
        tokens = next;
        history.insert(history.end(), next.begin(), next.end());
        if (round + 1 == first_steps) {
            windows[0].seconds = seconds_since(start);
            windows[0].counts  = forward.expert_cache_counts();
        }
    }
    windows[1].seconds = seconds_since(start);
    if (profile) { CUDA_CHECK(cudaProfilerStop()); }
    windows[1].counts = forward.expert_cache_counts();
    return history;
}

int run_bench(const Options& options, std::span<const std::int32_t> stream_tokens) {
    std::int32_t longest = 0;
    for (const std::int32_t length : options.prompt_lengths) {
        if (length < 1) { throw std::invalid_argument("prompt lengths must be positive"); }
        if (static_cast<std::size_t>(length) > stream_tokens.size()) {
            throw std::invalid_argument("prompt length " + std::to_string(length) +
                                        " exceeds the token file's " +
                                        std::to_string(stream_tokens.size()) + " tokens");
        }
        longest = std::max(longest, length);
    }
    std::filesystem::create_directories(options.out);
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(options.artifact);
    const fn::LoadedModel model(reader, device);
    const std::vector<std::uint64_t> calib_counts = options.calib_counts.empty()
                                                        ? std::vector<std::uint64_t>{}
                                                        : read_counts(options.calib_counts);
    // The prompt, one warm-up round and the timed rounds, or --context when larger.
    const std::int64_t context =
        std::max<std::int64_t>(std::int64_t{longest} + 1 + options.decode_tokens, options.context);
    if (context > ninfer::ops::kQsaMaximumCapacity) {
        throw std::invalid_argument("prompt plus decode steps exceed the context capacity");
    }
    fn::Program program(model,
                        program_options_for(options, static_cast<std::int32_t>(context),
                                            fn::LogitColumns::Last, calib_counts),
                        device);
    print_plan(program);

    std::ofstream report(options.out / "bench.json");
    report << std::setprecision(6) << "[\n";
    for (std::size_t run_index = 0; run_index < options.prompt_lengths.size(); ++run_index) {
        const std::int32_t length = options.prompt_lengths[run_index];
        const auto prompt         = stream_tokens.first(static_cast<std::size_t>(length));
        const fn::Lane lane       = open_greedy(program, context);
        const auto prefill_start  = std::chrono::steady_clock::now();
        std::vector<std::int32_t> token{prefill_prompt(program, lane, prompt)};
        const double prefill_seconds = seconds_since(prefill_start);

        // One untimed round first: the first replay follows the prefill's cache churn.
        std::vector<std::int32_t> next(1);
        program.decode(std::span<const fn::Lane>(&lane, 1), token, next);
        token                          = next;
        const std::int32_t first_token = token[0];
        std::array<DecodeWindow, 2> windows{};
        const std::vector<std::int32_t> graph_tokens =
            timed_rounds(program, std::span<const fn::Lane>(&lane, 1), token, options.decode_tokens,
                         options.profile_decode, windows);
        program.close_lane(lane);

        // The graph replays the eager round's kernels on the same inputs, so greedy continuations
        // from the same prompt must agree token for token (expert-cache contents change only where
        // a record is read from, never its values). Checked on the first prompt length: the count
        // of agreeing timed rounds, -2 when the warm-up token already differs, -1 when unchecked.
        std::int32_t agreeing = -1;
        if (run_index == 0) {
            const fn::Lane eager_lane = open_greedy(program, context);
            const std::array<fn::Lane, 1> lanes{eager_lane};
            const std::vector<ninfer::ops::SamplingConfig> configs = greedy_configs(1);
            std::int32_t eager = prefill_prompt(program, eager_lane, prompt);
            const auto step    = [&](std::int32_t input) {
                return program.forward()
                    .decode(lanes, std::span<const std::int32_t>(&input, 1), configs,
                            fn::RoundLogprobs::Skip, fn::DecodeExecution::Eager)
                    .tokens[0];
            };
            eager    = step(eager);
            agreeing = eager == first_token ? 0 : -2;
            for (std::size_t s = 0; agreeing >= 0 && s < graph_tokens.size(); ++s) {
                eager = step(eager);
                if (eager != graph_tokens[s]) { break; }
                ++agreeing;
            }
            program.close_lane(eager_lane);
            std::cerr << "graph vs eager greedy decode: " << agreeing << " of "
                      << graph_tokens.size() << " tokens agree\n";
        }

        std::cerr << "prompt " << length << ": prefill " << length / prefill_seconds << " tok/s\n";
        report << "  {\"prompt_tokens\": " << length
               << ", \"prefill_tokens_per_second\": " << length / prefill_seconds;
        for (std::size_t w = 0; w < windows.size(); ++w) {
            const char* name = w == 0 ? "first" : "whole";
            std::cerr << "  " << name << ' ';
            report << ", \"" << name << "\": ";
            windows[w].report(std::cerr, report);
        }
        report << ", \"graph_eager_agreeing_tokens\": " << agreeing << "}"
               << (run_index + 1 < options.prompt_lengths.size() ? ",\n" : "\n");
    }
    report << "]\n";
    if (!report) { throw std::runtime_error("cannot write bench.json"); }
    return 0;
}

template <class T>
std::vector<T> download(const ninfer::Tensor& tensor, cudaStream_t stream) {
    std::vector<T> host(static_cast<std::size_t>(tensor.bytes()) / sizeof(T));
    CUDA_CHECK(
        cudaMemcpyAsync(host.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}

template <class T>
void write_npy(const std::filesystem::path& path, std::string_view descr,
               std::span<const std::int64_t> shape, std::span<const T> values) {
    fnt::NpyWriter writer(path, descr, shape);
    writer.write(std::as_bytes(values));
}

// Writes the expert cache's state (the starting point of a policy replay) into `dir`: per pool c
// owner_p<c>.npy and referenced_p<c>.npy, clock_hands.npy, and every expert's codec codecs.npy.
void dump_cache_state(fn::Program& program, const std::filesystem::path& dir,
                      ninfer::DeviceContext& device) {
    const ninfer::ops::MoeExpertCache cache = program.experts().cache();
    for (std::int32_t pool = 0; pool < ninfer::ops::kMoeCacheCodecTags; ++pool) {
        const ninfer::ops::MoeExpertSlotPool& slots = cache.pools[pool];
        const std::string tag                       = std::to_string(pool);
        const std::vector<std::int32_t> owner =
            slots.owner.data == nullptr ? std::vector<std::int32_t>{}
                                        : download<std::int32_t>(slots.owner, device.stream);
        const std::vector<std::uint8_t> referenced =
            slots.referenced.data == nullptr
                ? std::vector<std::uint8_t>{}
                : download<std::uint8_t>(slots.referenced, device.stream);
        const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(owner.size())};
        write_npy(dir / ("owner_p" + tag + ".npy"), "<i4", shape,
                  std::span<const std::int32_t>(owner));
        write_npy(dir / ("referenced_p" + tag + ".npy"), "|u1", shape,
                  std::span<const std::uint8_t>(referenced));
    }
    const auto frequency = download<std::int32_t>(cache.frequency.counts, device.stream);
    const auto lookups   = download<std::int32_t>(cache.frequency.lookups, device.stream);
    write_npy(dir / "frequency.npy", "<i4",
              std::array<std::int64_t, 2>{fn::kLayers, ninfer::ops::kMoeExperts},
              std::span<const std::int32_t>(frequency));
    write_npy(dir / "frequency_lookups.npy", "<i4", std::array<std::int64_t, 1>{1},
              std::span<const std::int32_t>(lookups));
    const std::vector<std::int32_t> hands =
        download<std::int32_t>(cache.clock_hands, device.stream);
    write_npy(dir / "clock_hands.npy", "<i4",
              std::array<std::int64_t, 1>{ninfer::ops::kMoeCacheCodecTags},
              std::span<const std::int32_t>(hands));
    const std::vector<std::uint64_t> host_table =
        download<std::uint64_t>(program.experts().host_table(), device.stream);
    std::vector<std::uint8_t> codecs(host_table.size());
    std::ranges::transform(host_table, codecs.begin(), [](std::uint64_t address) {
        return static_cast<std::uint8_t>(address & 1U);
    });
    write_npy(dir / "codecs.npy", "|u1",
              std::array<std::int64_t, 2>{static_cast<std::int64_t>(codecs.size()) /
                                              ninfer::ops::kMoeExperts,
                                          ninfer::ops::kMoeExperts},
              std::span<const std::uint8_t>(codecs));
}

int run_trace(const Options& options, std::span<const std::int32_t> stream_tokens) {
    const std::int32_t length = options.prompt_lengths.front();
    const std::size_t needed =
        static_cast<std::size_t>(length) +
        (options.teacher ? static_cast<std::size_t>(options.decode_tokens) : 0);
    if (length < 1 || needed > stream_tokens.size()) {
        throw std::invalid_argument(
            "trace mode: the prompt (and with --teacher its continuation) exceeds the tokens");
    }
    std::filesystem::create_directories(options.out);
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(options.artifact);
    const fn::LoadedModel model(reader, device);
    const std::vector<std::uint64_t> calib_counts = options.calib_counts.empty()
                                                        ? std::vector<std::uint64_t>{}
                                                        : read_counts(options.calib_counts);
    const std::int64_t context =
        std::max<std::int64_t>(std::int64_t{length} + options.decode_tokens, options.context);
    if (context > ninfer::ops::kQsaMaximumCapacity) {
        throw std::invalid_argument("prompt plus decode steps exceed the context capacity");
    }
    fn::Program program(model,
                        program_options_for(options, static_cast<std::int32_t>(context),
                                            fn::LogitColumns::Last, calib_counts),
                        device);
    print_plan(program);
    fn::Forward& forward = program.forward();
    const fn::Lane lane  = open_greedy(program, context);
    std::int32_t token =
        prefill_prompt(program, lane, stream_tokens.first(static_cast<std::size_t>(length)));

    dump_cache_state(program, options.out, device);

    // Eager single-token prefill steps: the decode round's routed-expert and cache transitions at
    // T = 1, with the routes observable after each layer's moe_route.
    fnt::NpyWriter routes(
        options.out / "routes.npy", "<i4",
        std::array<std::int64_t, 3>{options.decode_tokens, fn::kLayers, ninfer::ops::kMoeTopK});
    std::vector<std::int32_t> step_routes(static_cast<std::size_t>(fn::kLayers) *
                                          ninfer::ops::kMoeTopK);
    fn::ForwardObserver observer;
    observer.after_route = [&](std::int32_t layer, const ninfer::Tensor& ids) {
        const std::vector<std::int32_t> routed = download<std::int32_t>(ids, device.stream);
        std::ranges::copy(routed, step_routes.begin() +
                                      static_cast<std::ptrdiff_t>(layer) * ninfer::ops::kMoeTopK);
    };
    std::ofstream report(options.out / "trace.json");
    const fn::ExpertSlotPlan& plan = program.experts().plan();
    report << "{\"prompt_tokens\": " << length << ", \"nvfp4_slots\": " << plan.slots[0]
           << ", \"f3_slots\": " << plan.slots[1] << ", \"bucket_steps\": " << options.bucket
           << ", \"buckets\": [\n";
    forward.clear_expert_cache_counts();
    for (std::int32_t step = 0; step < options.decode_tokens; ++step) {
        if (options.teacher) {
            token =
                stream_tokens[static_cast<std::size_t>(length) + static_cast<std::size_t>(step)];
        }
        forward.prefill(lane, std::span<const std::int32_t>(&token, 1), fn::LogitsSink{},
                        &observer);
        token = forward.sample_last(ninfer::ops::SamplingConfig{}, program.committed(lane) - 1)
                    .tokens[0];
        routes.write(std::as_bytes(std::span<const std::int32_t>(step_routes)));
        const bool last = step + 1 == options.decode_tokens;
        if ((step + 1) % options.bucket != 0 && !last) { continue; }
        const fn::ExpertCacheCounts counts = forward.expert_cache_counts();
        forward.clear_expert_cache_counts();
        const std::int64_t lookups = counts.lookups[0] + counts.lookups[1];
        const std::int64_t misses  = counts.misses[0] + counts.misses[1];
        report << "  {\"end_step\": " << step + 1 << ", \"lookups\": [" << counts.lookups[0] << ", "
               << counts.lookups[1] << "], \"misses\": [" << counts.misses[0] << ", "
               << counts.misses[1] << "]}" << (last ? "\n" : ",\n");
        std::cerr << "steps " << step + 1 << ": hit rate "
                  << 1.0 - static_cast<double>(misses) / static_cast<double>(lookups) << '\n';
    }
    report << "]}\n";
    if (!report) { throw std::runtime_error("cannot write trace.json"); }
    return 0;
}

// FNV-1a over a BF16 column: equality of logits across runs without keeping them.
std::uint64_t fingerprint(std::span<const std::uint16_t> values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint16_t value : values) { hash = (hash ^ value) * 1099511628211ULL; }
    return hash;
}

// Per-step comparisons of batched against solo logits, summarized.
struct LaneComparison {
    // Rows per round: kls holds round r's rows at [r rows, (r + 1) rows).
    std::int32_t rows = 1;
    std::vector<double> kls;
    std::int64_t top1_agree = 0;

    void add(const Comparison& c) {
        kls.push_back(c.kl);
        top1_agree += c.top1_agree ? 1 : 0;
    }

    void report(std::ostream& log, std::ostream& json) const {
        std::vector<double> sorted = kls;
        std::ranges::sort(sorted);
        double mean = 0.0;
        for (const double kl : sorted) { mean += kl; }
        mean /= static_cast<double>(sorted.size());
        const auto quantile = [&](double q) {
            return sorted[static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1))];
        };
        // Mean KL per bucket of kBucketRounds rounds: drift over the decode.
        constexpr std::size_t kBucketRounds = 64;
        const std::size_t bucket            = kBucketRounds * static_cast<std::size_t>(rows);
        std::vector<double> buckets;
        for (std::size_t first = 0; first < kls.size(); first += bucket) {
            const std::size_t last = std::min(kls.size(), first + bucket);
            double sum             = 0.0;
            for (std::size_t k = first; k < last; ++k) { sum += kls[k]; }
            buckets.push_back(sum / static_cast<double>(last - first));
        }
        log << sorted.size() << " steps: KL mean " << mean << ", p50 " << quantile(0.5) << ", p99 "
            << quantile(0.99) << ", max " << sorted.back() << "; top-1 " << top1_agree << " / "
            << sorted.size() << "; mean per 64 rounds";
        json << "{\"steps\": " << sorted.size() << ", \"kl_mean\": " << mean
             << ", \"kl_p50\": " << quantile(0.5) << ", \"kl_p99\": " << quantile(0.99)
             << ", \"kl_max\": " << sorted.back() << ", \"top1_agree\": " << top1_agree
             << ", \"kl_mean_per_64_rounds\": [";
        for (std::size_t b = 0; b < buckets.size(); ++b) {
            log << ' ' << buckets[b];
            json << (b == 0 ? "" : ", ") << buckets[b];
        }
        log << '\n';
        json << "]}";
    }
};

// lanes-check: per-lane correctness of lane-batched decode. Lane i's prompt is tokens [i P,
// (i + 1) P) of the token stream (P = the first --prompt-lengths value). Each lane is first run
// solo (B = 1, graph): its prefill, then --decode greedy rounds, keeping every round's input token
// and logits. Every batch size B of --lanes then reruns lanes 0 .. B-1 together, teacher-forced
// on the solo inputs, through the graphs and eagerly; each round's column of lane i is compared
// with lane i's solo logits (KL(solo || batched), top-1), and the eager logits with the graph's
// bit for bit. With --interleave N, the B = 2 graph run is repeated while a third lane prefills N
// tokens of --interleave-tokens one super-chunk every eight rounds (a fourth lane's short prompt
// interrupting it once), and must reproduce the B = 2 logits bit for bit; the interleaved lane's
// first token and logits must equal those of an uninterrupted prefill of the same prompt.
//
// The solo logits are kept on the host: --decode rounds x 248320 x 2 bytes per lane (1.5 GB at 256
// rounds and 12 lanes).
int run_lanes_check(const Options& options, std::span<const std::int32_t> stream_tokens) {
    // The interrupting prompt of the interleave check.
    constexpr std::int32_t kShortPromptTokens = 1024;
    const std::int32_t prompt_tokens          = options.prompt_lengths.front();
    const std::int32_t rounds                 = options.decode_tokens;
    const std::int32_t lanes_max              = std::ranges::max(options.lane_counts);
    if (std::ranges::min(options.lane_counts) < 1 || lanes_max > fn::kMaxLanes) {
        throw std::invalid_argument("--lanes: batch sizes in [1, 12]");
    }
    if (options.interleave_tokens > 0 &&
        std::ranges::find(options.lane_counts, 2) == options.lane_counts.end()) {
        throw std::invalid_argument("--interleave compares with the B=2 run: --lanes must hold 2");
    }
    const std::int32_t solo_lanes = lanes_max;
    if (static_cast<std::size_t>(prompt_tokens) * static_cast<std::size_t>(solo_lanes) >
        stream_tokens.size()) {
        throw std::invalid_argument("lanes-check: the token file holds too few prompts");
    }
    std::vector<std::int32_t> interleave;
    if (options.interleave_tokens > 0) {
        interleave = read_tokens(options.interleave_file, 1).tokens;
        if (interleave.size() <
            static_cast<std::size_t>(options.interleave_tokens) + kShortPromptTokens) {
            throw std::invalid_argument("--interleave-tokens holds too few tokens");
        }
        interleave.resize(static_cast<std::size_t>(options.interleave_tokens) + kShortPromptTokens);
    }
    const std::int64_t reserve = std::int64_t{prompt_tokens} + rounds + 1;
    const std::int64_t context = std::max<std::int64_t>(
        {reserve, options.interleave_tokens + 1, kShortPromptTokens + 1, options.context});
    std::filesystem::create_directories(options.out);
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(options.artifact);
    const fn::LoadedModel model(reader, device);
    const std::vector<std::uint64_t> calib_counts = options.calib_counts.empty()
                                                        ? std::vector<std::uint64_t>{}
                                                        : read_counts(options.calib_counts);
    const std::int32_t program_lanes = std::max(lanes_max, options.interleave_tokens > 0 ? 4 : 1);
    fn::Program program(model,
                        program_options_for(options, static_cast<std::int32_t>(context),
                                            fn::LogitColumns::Last, calib_counts, program_lanes),
                        device);
    print_plan(program);
    fn::Forward& forward  = program.forward();
    constexpr auto kVocab = static_cast<std::size_t>(fn::kVocabRows);
    const auto prompt     = [&](std::int32_t lane) {
        return stream_tokens.subspan(static_cast<std::size_t>(lane) * prompt_tokens,
                                     static_cast<std::size_t>(prompt_tokens));
    };

    // Solo references.
    std::vector<std::int32_t> first_tokens(static_cast<std::size_t>(solo_lanes));
    std::vector<std::vector<std::int32_t>> inputs(static_cast<std::size_t>(solo_lanes));
    std::vector<std::vector<std::uint16_t>> solo(static_cast<std::size_t>(solo_lanes));
    for (std::int32_t i = 0; i < solo_lanes; ++i) {
        const fn::Lane lane                       = open_greedy(program, reserve);
        std::int32_t token                        = prefill_prompt(program, lane, prompt(i));
        first_tokens[static_cast<std::size_t>(i)] = token;
        std::vector<std::uint16_t>& logits        = solo[static_cast<std::size_t>(i)];
        logits.reserve(static_cast<std::size_t>(rounds) * kVocab);
        for (std::int32_t r = 0; r < rounds; ++r) {
            inputs[static_cast<std::size_t>(i)].push_back(token);
            std::int32_t next = 0;
            program.decode(std::span<const fn::Lane>(&lane, 1),
                           std::span<const std::int32_t>(&token, 1),
                           std::span<std::int32_t>(&next, 1));
            const std::vector<std::uint16_t> host = download_bf16(forward.logits(1), device.stream);
            logits.insert(logits.end(), host.begin(), host.end());
            token = next;
        }
        program.close_lane(lane);
        std::cerr << "solo lane " << i << " done\n";
    }

    // Runs lanes 0 .. rows-1 teacher-forced on the solo inputs; per round and row, the logits'
    // fingerprint and, when `comparison` is given, the comparison with solo. `between` runs after
    // every round.
    const auto batched = [&](std::int32_t rows, fn::DecodeExecution execution,
                             LaneComparison* comparison, std::int32_t& first_agree,
                             const std::function<void(std::int32_t)>& between) {
        std::vector<fn::Lane> lanes;
        first_agree = 0;
        for (std::int32_t i = 0; i < rows; ++i) {
            lanes.push_back(open_greedy(program, reserve));
            first_agree += prefill_prompt(program, lanes.back(), prompt(i)) ==
                                   first_tokens[static_cast<std::size_t>(i)]
                               ? 1
                               : 0;
        }
        const std::vector<ninfer::ops::SamplingConfig> configs =
            greedy_configs(static_cast<std::size_t>(rows));
        std::vector<std::uint64_t> fingerprints;
        std::vector<std::int32_t> tokens(static_cast<std::size_t>(rows));
        std::vector<std::int32_t> next(static_cast<std::size_t>(rows));
        for (std::int32_t r = 0; r < rounds; ++r) {
            for (std::int32_t i = 0; i < rows; ++i) {
                tokens[static_cast<std::size_t>(i)] =
                    inputs[static_cast<std::size_t>(i)][static_cast<std::size_t>(r)];
            }
            if (execution == fn::DecodeExecution::Graph) {
                program.decode(lanes, tokens, next);
            } else {
                static_cast<void>(
                    forward.decode(lanes, tokens, configs, fn::RoundLogprobs::Skip, execution));
            }
            const std::vector<std::uint16_t> host =
                download_bf16(forward.logits(rows), device.stream);
            for (std::int32_t i = 0; i < rows; ++i) {
                const std::span<const std::uint16_t> column(
                    host.data() + static_cast<std::size_t>(i) * kVocab, kVocab);
                fingerprints.push_back(fingerprint(column));
                if (comparison == nullptr) { continue; }
                const std::span<const std::uint16_t> reference(
                    solo[static_cast<std::size_t>(i)].data() + static_cast<std::size_t>(r) * kVocab,
                    kVocab);
                comparison->add(compare(reference, column));
            }
            if (between) { between(r); }
        }
        for (const fn::Lane lane : lanes) { program.close_lane(lane); }
        return fingerprints;
    };

    std::ofstream report(options.out / "lanes_check.json");
    report << std::setprecision(6) << "{\"prompt_tokens\": " << prompt_tokens
           << ", \"rounds\": " << rounds << ", \"batches\": [\n";
    std::vector<std::uint64_t> two_lane_graph;
    for (std::size_t index = 0; index < options.lane_counts.size(); ++index) {
        const std::int32_t rows = options.lane_counts[index];
        LaneComparison graph_cmp{.rows = rows};
        std::int32_t graph_first = 0;
        std::int32_t eager_first = 0;
        const std::vector<std::uint64_t> graph =
            batched(rows, fn::DecodeExecution::Graph, &graph_cmp, graph_first, {});
        const std::vector<std::uint64_t> eager =
            batched(rows, fn::DecodeExecution::Eager, nullptr, eager_first, {});
        if (rows == 2) { two_lane_graph = graph; }
        std::int64_t identical = 0;
        for (std::size_t k = 0; k < graph.size(); ++k) {
            identical += graph[k] == eager[k] ? 1 : 0;
        }
        std::cerr << "B=" << rows << " graph vs solo: first tokens " << graph_first << " / " << rows
                  << ", ";
        report << "  {\"lanes\": " << rows << ", \"first_tokens_agree\": " << graph_first
               << ", \"graph_vs_solo\": ";
        graph_cmp.report(std::cerr, report);
        std::cerr << "B=" << rows << " graph vs eager: " << identical << " / " << graph.size()
                  << " columns bit-identical\n";
        report << ", \"graph_eager_identical_columns\": " << identical
               << ", \"columns\": " << graph.size() << "}"
               << (index + 1 < options.lane_counts.size() ? ",\n" : "\n");
    }
    report << "]";

    // The noise floor of an equally valid route: lanes 0 and 1 teacher-forced through single-token
    // prefill steps (the prefill GDN path: one projection, then the in-place convolution and
    // recurrence) against their solo decode rounds (the snapshot path).
    {
        LaneComparison route{.rows = 1};
        for (std::int32_t i = 0; i < std::min(2, solo_lanes); ++i) {
            const fn::Lane lane = open_greedy(program, reserve);
            static_cast<void>(prefill_prompt(program, lane, prompt(i)));
            for (std::int32_t r = 0; r < rounds; ++r) {
                const std::int32_t input =
                    inputs[static_cast<std::size_t>(i)][static_cast<std::size_t>(r)];
                forward.prefill(lane, std::span<const std::int32_t>(&input, 1), fn::LogitsSink{});
                const std::vector<std::uint16_t> host =
                    download_bf16(forward.logits(1), device.stream);
                route.add(compare(
                    std::span<const std::uint16_t>(solo[static_cast<std::size_t>(i)].data() +
                                                       static_cast<std::size_t>(r) * kVocab,
                                                   kVocab),
                    host));
            }
            program.close_lane(lane);
        }
        std::cerr << "single-token prefill route vs solo decode: ";
        report << ", \"prefill_route_vs_solo\": ";
        route.report(std::cerr, report);
    }

    if (options.interleave_tokens > 0) {
        const std::span<const std::int32_t> long_prompt =
            std::span<const std::int32_t>(interleave)
                .first(static_cast<std::size_t>(options.interleave_tokens));
        const std::span<const std::int32_t> short_prompt =
            std::span<const std::int32_t>(interleave)
                .subspan(static_cast<std::size_t>(options.interleave_tokens), kShortPromptTokens);
        // Reference: the long prompt prefilled without interruption.
        const fn::Lane reference_lane      = open_greedy(program, options.interleave_tokens + 1);
        const std::int32_t reference_token = prefill_prompt(program, reference_lane, long_prompt);
        const std::uint64_t reference_logits =
            fingerprint(download_bf16(forward.logits(1), device.stream));
        program.close_lane(reference_lane);

        std::optional<fn::Lane> long_lane;
        std::span<const std::int32_t> remaining = long_prompt;
        std::int32_t steps                      = 0;
        std::int32_t interleaved_token          = -1;
        std::uint64_t interleaved_logits        = 0;
        const auto between                      = [&](std::int32_t round) {
            if (round % 8 != 7 || remaining.empty()) { return; }
            if (!long_lane) { long_lane = open_greedy(program, options.interleave_tokens + 1); }
            const fn::PrefillStep step = program.prefill_step(*long_lane, remaining);
            remaining                  = remaining.subspan(static_cast<std::size_t>(step.consumed));
            if (step.complete) {
                interleaved_token  = step.next_token;
                interleaved_logits = fingerprint(download_bf16(forward.logits(1), device.stream));
            }
            if (++steps == 1) {
                // A short prompt interrupts the long one between its super-chunks.
                const fn::Lane short_lane = open_greedy(program, kShortPromptTokens + 1);
                static_cast<void>(prefill_prompt(program, short_lane, short_prompt));
                program.close_lane(short_lane);
            }
        };
        std::int32_t first_agree = 0;
        const std::vector<std::uint64_t> fingerprints =
            batched(2, fn::DecodeExecution::Graph, nullptr, first_agree, between);
        if (long_lane) { program.close_lane(*long_lane); }
        std::int64_t identical = 0;
        for (std::size_t k = 0; k < fingerprints.size() && k < two_lane_graph.size(); ++k) {
            identical += fingerprints[k] == two_lane_graph[k] ? 1 : 0;
        }
        const bool prompt_done = remaining.empty();
        std::cerr << "interleaved prefill (" << steps << " super-chunks, complete " << prompt_done
                  << "): B=2 columns bit-identical " << identical << " / " << fingerprints.size()
                  << "; interleaved lane first token "
                  << (interleaved_token == reference_token ? "agrees" : "differs") << ", logits "
                  << (interleaved_logits == reference_logits ? "identical" : "differ") << '\n';
        report << ", \"interleave\": {\"prompt_tokens\": " << options.interleave_tokens
               << ", \"super_chunks\": " << steps << ", \"complete\": " << prompt_done
               << ", \"identical_columns\": " << identical
               << ", \"columns\": " << fingerprints.size()
               << ", \"first_token_agrees\": " << (interleaved_token == reference_token)
               << ", \"logits_identical\": " << (interleaved_logits == reference_logits) << "}";
    }
    report << "}\n";
    if (!report) { throw std::runtime_error("cannot write lanes_check.json"); }
    return 0;
}

#include "lanes_bench.h"
#include "dflash_acceptance.h"

// lanes-trace: the expert-cache trace of lane-batched decode, for policy replay. Per concurrency C
// of --lanes (each with its own Program and shared QSA pool), C lanes prefilled with prompts
// i + --prompt-offset, then --decode greedy rounds run through eager plain decode or captured
// fixed-K native verification, writing into <out>/c<C>/:
//   owner_p<c>.npy, referenced_p<c>.npy, clock_hands.npy, codecs.npy  post-prefill cache state
//   routes.npy   int32 [rounds,48,C,W,10]: routed expert ids per layer, lane and column
//   frequency.npy, frequency_lookups.npy  exact post-prefill admission-filter state
//   trace.json   slot counts and, per bucket of --bucket rounds, the device counters' lookups and
//                misses per codec
//   final/       the same cache and frequency state after the last round
int run_lanes_trace(const Options& options, std::span<const std::int32_t> stream_tokens) {
    const std::int32_t prompt_tokens = options.prompt_lengths.front();
    const std::int32_t rounds        = options.decode_tokens;
    const bool speculative = options.speculative.backend != ninfer::SpeculativeBackend::None;
    if (options.speculative.adaptive_draft) {
        throw std::invalid_argument("lanes-trace requires a fixed verification width");
    }
    const auto width =
        speculative ? static_cast<std::int32_t>(options.speculative.draft_tokens) + 1 : 1;
    const std::int64_t reserve = std::int64_t{prompt_tokens} + std::int64_t{rounds} * width + 1;
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(options.artifact);
    const fn::LoadedModel model(reader, device);
    const std::vector<std::uint64_t> calib_counts = options.calib_counts.empty()
                                                        ? std::vector<std::uint64_t>{}
                                                        : read_counts(options.calib_counts);
    for (const std::int32_t lanes : options.lane_counts) {
        if (lanes < 1 || lanes > fn::kMaxLanes ||
            static_cast<std::size_t>(lanes + options.prompt_offset) *
                    static_cast<std::size_t>(prompt_tokens) >
                stream_tokens.size()) {
            throw std::invalid_argument("lanes-trace: concurrency outside [1, 12] or too few "
                                        "prompt tokens");
        }
        const std::int64_t context = options.kv_capacity
                                         ? std::min<std::int64_t>(*options.kv_capacity / lanes,
                                                                  ninfer::ops::kQsaMaximumCapacity)
                                         : std::max<std::int64_t>(reserve, options.context);
        if (context < reserve) {
            throw std::invalid_argument("lanes-trace: the lane context is below the prompt and "
                                        "the decode rounds");
        }
        const std::filesystem::path dir = options.out / ("c" + std::to_string(lanes));
        std::filesystem::create_directories(dir);
        fn::Program program(model,
                            program_options_for(options, static_cast<std::int32_t>(context),
                                                fn::LogitColumns::Last, calib_counts, lanes,
                                                options.kv_capacity),
                            device);
        print_plan(program);
        fn::Forward& forward = program.forward();
        std::vector<fn::Lane> open;
        std::vector<std::int32_t> tokens;
        for (std::int32_t i = 0; i < lanes; ++i) {
            open.push_back(open_greedy(program, reserve));
            tokens.push_back(prefill_prompt(
                program, open.back(),
                stream_tokens.subspan(static_cast<std::size_t>(i + options.prompt_offset) *
                                          static_cast<std::size_t>(prompt_tokens),
                                      static_cast<std::size_t>(prompt_tokens))));
        }
        dump_cache_state(program, dir, device);

        const auto per_layer = static_cast<std::size_t>(lanes) * static_cast<std::size_t>(width) *
                               ninfer::ops::kMoeTopK;
        fnt::NpyWriter routes(
            dir / "routes.npy", "<i4",
            std::array<std::int64_t, 5>{rounds, fn::kLayers, lanes, width, ninfer::ops::kMoeTopK});
        std::vector<std::int32_t> round_routes(static_cast<std::size_t>(fn::kLayers) * per_layer);
        fn::ForwardObserver observer;
        observer.after_route = [&](std::int32_t layer, const ninfer::Tensor& ids) {
            // ids is I32 [10,C]: the k-th expert of lane i at element 10 i + k.
            const std::vector<std::int32_t> routed = download<std::int32_t>(ids, device.stream);
            std::ranges::copy(routed,
                              round_routes.begin() + static_cast<std::ptrdiff_t>(layer) *
                                                         static_cast<std::ptrdiff_t>(per_layer));
        };
        const std::vector<ninfer::ops::SamplingConfig> configs =
            greedy_configs(static_cast<std::size_t>(lanes));
        const fn::ExpertSlotPlan& plan = program.experts().plan();
        std::ofstream report(dir / "trace.json");
        report << "{\"prompt_tokens\": " << prompt_tokens << ", \"lanes\": " << lanes
               << ", \"width\": " << width
               << ", \"kv_capacity_tokens\": " << program.kv_capacity_tokens()
               << ", \"nvfp4_slots\": " << plan.slots[0] << ", \"f3_slots\": " << plan.slots[1]
               << ", \"f3_flex_slots\": " << program.experts().flex_slots()
               << ", \"f3_flex_lent_slots\": "
               << program.extents().lent_extents() * program.extents().slots_per_extent()
               << ", \"bucket_steps\": " << options.bucket << ", \"buckets\": [\n";
        forward.clear_expert_cache_counts();
        std::uint64_t published_tokens = 0;
        const std::vector<std::uint32_t> limits(static_cast<std::size_t>(lanes),
                                                static_cast<std::uint32_t>(width));
        for (std::int32_t round = 0; round < rounds; ++round) {
            if (speculative) {
                const auto packet =
                    program.decode_round(open, tokens, limits, fn::RoundLogprobs::Skip);
                for (std::int32_t layer = 0; layer < fn::kLayers; ++layer) {
                    const auto routed =
                        download<std::int32_t>(forward.verify_routes(layer), device.stream);
                    std::ranges::copy(routed, round_routes.begin() +
                                                  static_cast<std::ptrdiff_t>(layer) *
                                                      static_cast<std::ptrdiff_t>(per_layer));
                }
                for (std::size_t row = 0; row < tokens.size(); ++row) {
                    tokens[row] = packet.tokens[row * static_cast<std::size_t>(width) +
                                                static_cast<std::size_t>(packet.counts[row] - 1)];
                }
                for (const auto count : packet.counts) {
                    published_tokens += static_cast<std::uint32_t>(count);
                }
                program.resolve_round(packet.counts);
            } else {
                const fn::SampledRows rows =
                    forward.decode(open, tokens, configs, fn::RoundLogprobs::Skip,
                                   fn::DecodeExecution::Eager, &observer);
                std::ranges::copy(rows.tokens, tokens.begin());
                published_tokens += static_cast<std::uint32_t>(lanes);
            }
            routes.write(std::as_bytes(std::span<const std::int32_t>(round_routes)));
            const bool last = round + 1 == rounds;
            if ((round + 1) % options.bucket != 0 && !last) { continue; }
            const fn::ExpertCacheCounts counts = forward.expert_cache_counts();
            forward.clear_expert_cache_counts();
            report << "  {\"end_step\": " << round + 1 << ", \"lookups\": [" << counts.lookups[0]
                   << ", " << counts.lookups[1] << "], \"misses\": [" << counts.misses[0] << ", "
                   << counts.misses[1] << "]}" << (last ? "\n" : ",\n");
        }
        report << "],\"published_tokens\":" << published_tokens << "}\n";
        std::filesystem::create_directories(dir / "final");
        dump_cache_state(program, dir / "final", device);
        if (!report) { throw std::runtime_error("cannot write trace.json"); }
        std::cerr << "C=" << lanes << ": traced " << rounds << " rounds\n";
    }
    return 0;
}

// Teacher-forced decode panel: writes logprobs.npy like prefill mode's --logits all, with
// position 0 from a one-token prefill and positions 1 .. T-2 from graph-replayed B=1 decode steps
// whose input is the file's token. Position p's logits predict token p+1.
int run_decode_forced(const Options& options, std::span<const std::int32_t> all_tokens,
                      std::int64_t sequences) {
    const auto length =
        static_cast<std::int32_t>(all_tokens.size() / static_cast<std::size_t>(sequences));
    if (length < 2) { throw std::invalid_argument("decode-forced mode needs T >= 2"); }
    std::filesystem::create_directories(options.out);
    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(options.artifact);
    const fn::LoadedModel model(reader, device);
    const std::vector<std::uint64_t> calib_counts = options.calib_counts.empty()
                                                        ? std::vector<std::uint64_t>{}
                                                        : read_counts(options.calib_counts);
    fn::Program program(model,
                        program_options_for(options, options.context > 0 ? options.context : length,
                                            fn::LogitColumns::Last, calib_counts),
                        device);
    print_plan(program);
    fn::Forward& forward = program.forward();

    const std::array<std::int64_t, 3> shape{sequences, length - 1, fn::kVocabRows};
    fnt::NpyWriter logprobs(options.out / "logprobs.npy", "<f2", shape);
    const auto write_position = [&](std::span<const std::uint16_t> logits) {
        const std::vector<std::uint16_t> row = log_softmax_half(logits);
        logprobs.write(std::as_bytes(std::span<const std::uint16_t>(row)));
    };
    for (std::int64_t s = 0; s < sequences; ++s) {
        const std::span<const std::int32_t> tokens(
            all_tokens.data() + static_cast<std::size_t>(s) * static_cast<std::size_t>(length),
            static_cast<std::size_t>(length));
        const fn::Lane lane = open_greedy(program, length);
        // Program-level prefill, as lanes-check opens lanes, so decode sees the lane's committed
        // history; its sampled token is discarded in favor of the panel's.
        static_cast<void>(prefill_prompt(program, lane, tokens.first(1)));
        write_position(download_bf16(forward.logits(1), device.stream));
        for (std::int32_t p = 1; p + 1 < length; ++p) {
            const std::int32_t input = tokens[static_cast<std::size_t>(p)];
            std::int32_t sampled     = 0;
            program.decode(std::span<const fn::Lane>(&lane, 1),
                           std::span<const std::int32_t>(&input, 1),
                           std::span<std::int32_t>(&sampled, 1));
            write_position(download_bf16(forward.logits(1), device.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        program.close_lane(lane);
        std::cerr << "sequence " << s << ": " << length << " tokens\n";
    }
    std::ofstream report(options.out / "decode_forced.json");
    report << "{\"expert_helper_pci\": "
           << (options.expert_helper_pci ? "\"" + *options.expert_helper_pci + "\"" : "null")
           << ", \"sequences\": " << sequences << ", \"length\": " << length << "}\n";
    if (!report) { throw std::runtime_error("cannot write decode_forced.json"); }
    return 0;
}

int run(const Options& options) {
    const TokenFile token_file                  = read_tokens(options.tokens, options.sequences);
    const std::vector<std::int32_t>& all_tokens = token_file.tokens;
    const std::int64_t sequences                = token_file.sequences;
    if (sequences < 1 || all_tokens.size() % static_cast<std::size_t>(sequences) != 0) {
        throw std::invalid_argument("token count is not a multiple of the sequence count");
    }
    if (options.decode_forced_mode) { return run_decode_forced(options, all_tokens, sequences); }
    if (options.bench_mode) { return run_bench(options, all_tokens); }
    if (options.lanes_check_mode) { return run_lanes_check(options, all_tokens); }
    if (options.lanes_bench_mode) { return run_lanes_bench(options, all_tokens); }
    if (options.dflash_acceptance_mode) { return run_dflash_acceptance(options, token_file); }
    if (options.lanes_trace_mode) { return run_lanes_trace(options, all_tokens); }
    if (options.trace_mode) { return run_trace(options, all_tokens); }
    const auto length =
        static_cast<std::int32_t>(all_tokens.size() / static_cast<std::size_t>(sequences));
    if (!options.split_mode && options.all_logits && length < 2) {
        throw std::invalid_argument("--logits all needs sequences of at least two tokens");
    }
    if (options.split_mode && (options.split < 1 || options.split >= length)) {
        throw std::invalid_argument("--split must be in [1, T)");
    }
    std::filesystem::create_directories(options.out);

    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(options.artifact);
    const fn::LoadedModel model(reader, device);
    const std::vector<std::uint64_t> calib_counts = options.calib_counts.empty()
                                                        ? std::vector<std::uint64_t>{}
                                                        : read_counts(options.calib_counts);
    fn::Program program(model,
                        program_options_for(options, options.context > 0 ? options.context : length,
                                            options.split_mode || options.all_logits
                                                ? fn::LogitColumns::All
                                                : fn::LogitColumns::Last,
                                            calib_counts),
                        device);
    print_plan(program);
    fn::Forward& forward = program.forward();

    constexpr auto kVocab = static_cast<std::size_t>(fn::kVocabRows);
    std::unique_ptr<fnt::NpyWriter> logprobs;
    if (!options.split_mode && options.all_logits) {
        const std::array<std::int64_t, 3> shape{sequences, length - 1, fn::kVocabRows};
        logprobs = std::make_unique<fnt::NpyWriter>(options.out / "logprobs.npy", "<f2", shape);
    }
    for (std::int64_t s = 0; s < sequences; ++s) {
        const std::span<const std::int32_t> tokens(
            all_tokens.data() + static_cast<std::size_t>(s) * static_cast<std::size_t>(length),
            static_cast<std::size_t>(length));
        const std::string tag = "s" + std::to_string(s);
        if (!options.split_mode) {
            const auto logits_path = options.out / ("logits_" + tag + ".bf16");
            std::filesystem::remove(logits_path);
            const fn::LogitsSink sink = [&](const ninfer::Tensor& logits, std::int32_t position) {
                const std::vector<std::uint16_t> host = download_bf16(logits, device.stream);
                if (!logprobs) {
                    append(logits_path, host);
                    return;
                }
                // The last position predicts nothing in the sequence and is not written.
                const std::int64_t columns =
                    std::min<std::int64_t>(logits.ne[1], length - 1 - position);
                for (std::int64_t c = 0; c < columns; ++c) {
                    const std::vector<std::uint16_t> row =
                        log_softmax_half(std::span<const std::uint16_t>(
                            host.data() + static_cast<std::size_t>(c) * kVocab, kVocab));
                    logprobs->write(std::as_bytes(std::span<const std::uint16_t>(row)));
                }
            };
            fn::ForwardObserver observer;
            observer.after_layer = [&](std::int32_t layer, const ninfer::Tensor& residual,
                                       std::int32_t) {
                append(options.out / ("residual_" + tag + "_l" + std::to_string(layer) + ".bf16"),
                       download_bf16(residual, device.stream));
            };
            if (options.residuals) {
                for (std::int32_t layer = 0; layer < fn::kLayers; ++layer) {
                    std::filesystem::remove(
                        options.out / ("residual_" + tag + "_l" + std::to_string(layer) + ".bf16"));
                }
            }
            const fn::Lane lane = open_greedy(program, length);
            forward.prefill(lane, tokens, sink, options.residuals ? &observer : nullptr);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            program.close_lane(lane);
            std::cerr << "sequence " << s << ": " << length << " tokens\n";
            continue;
        }

        // Reference: teacher-forced prefill of all T tokens, keeping positions T-K..T-1.
        const std::int32_t first = length - options.split;
        std::vector<std::uint16_t> reference(static_cast<std::size_t>(options.split) * kVocab);
        const fn::Lane reference_lane = open_greedy(program, length);
        forward.prefill(
            reference_lane, tokens, [&](const ninfer::Tensor& logits, std::int32_t position) {
                const std::vector<std::uint16_t> host = download_bf16(logits, device.stream);
                const auto columns                    = static_cast<std::int32_t>(logits.ne[1]);
                for (std::int32_t c = 0; c < columns; ++c) {
                    const std::int32_t p = position + c;
                    if (p < first) { continue; }
                    std::copy_n(host.begin() + static_cast<std::ptrdiff_t>(c) * kVocab, kVocab,
                                reference.begin() +
                                    static_cast<std::ptrdiff_t>(p - first) * kVocab);
                }
            });
        // Candidate: prefill of T-K tokens, then K single-token steps.
        std::vector<std::uint16_t> stepped(static_cast<std::size_t>(options.split) * kVocab);
        const fn::LogitsSink step_sink = [&](const ninfer::Tensor& logits, std::int32_t position) {
            const std::int32_t last = position + static_cast<std::int32_t>(logits.ne[1]) - 1;
            if (last < first) { return; }
            const std::vector<std::uint16_t> host = download_bf16(logits, device.stream);
            std::copy_n(host.end() - static_cast<std::ptrdiff_t>(kVocab), kVocab,
                        stepped.begin() + static_cast<std::ptrdiff_t>(last - first) * kVocab);
        };
        program.close_lane(reference_lane);
        const fn::Lane lane = open_greedy(program, length);
        forward.prefill(lane, tokens.first(static_cast<std::size_t>(first)), step_sink);
        for (std::int32_t p = first; p < length; ++p) {
            forward.prefill(lane, tokens.subspan(static_cast<std::size_t>(p), 1), step_sink);
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        program.close_lane(lane);

        std::ofstream report(options.out / ("split_" + tag + ".json"));
        report << std::setprecision(9) << "[\n";
        for (std::int32_t k = 0; k < options.split; ++k) {
            const std::span<const std::uint16_t> a(
                reference.data() + static_cast<std::size_t>(k) * kVocab, kVocab);
            const std::span<const std::uint16_t> b(
                stepped.data() + static_cast<std::size_t>(k) * kVocab, kVocab);
            const Comparison c = compare(a, b);
            report << "  {\"position\": " << first + k << ", \"max_abs\": " << c.max_abs
                   << ", \"mean_abs\": " << c.mean_abs << ", \"kl\": " << c.kl
                   << ", \"top1_agree\": " << (c.top1_agree ? "true" : "false") << "}"
                   << (k + 1 < options.split ? ",\n" : "\n");
            std::cerr << "sequence " << s << " position " << first + k << ": max |d| " << c.max_abs
                      << ", KL " << c.kl << ", top-1 " << (c.top1_agree ? "agrees" : "differs")
                      << '\n';
        }
        report << "]\n";
        if (!report) { throw std::runtime_error("cannot write the split report"); }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::span<char*> args(argv + 1, static_cast<std::size_t>(argc - 1));
        if (args.empty() || std::string_view(args[0]) == "--help") {
            std::cout << kUsage;
            return args.empty() ? 2 : 0;
        }
        return run(parse(args));
    } catch (const std::exception& error) {
        std::cerr << "qwen3_8_flash_next parity failed: " << error.what() << '\n';
        return 1;
    }
}
