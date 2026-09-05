#include "targets/qwen4/verifier.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;

namespace {

std::int32_t parse_width(std::string_view text) {
    std::int32_t width = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), width);
    if (error != std::errc{} || end != text.data() + text.size() || width <= 0 ||
        width > verifier::kMaximumPrefillChunk) {
        throw std::invalid_argument("width must be an integer in [1,4096]");
    }
    return width;
}

std::int32_t parse_repeats(std::string_view text) {
    std::int32_t repeats = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), repeats);
    if (error != std::errc{} || end != text.data() + text.size() || repeats <= 0 ||
        repeats > 20) {
        throw std::invalid_argument("repeats must be an integer in [1,20]");
    }
    return repeats;
}

std::int32_t sequence_token(std::int32_t position) {
    constexpr std::uint64_t vocabulary = 248320;
    std::uint64_t value = static_cast<std::uint64_t>(position) + 1U;
    value ^= value >> 12U;
    value ^= value << 25U;
    value ^= value >> 27U;
    value *= 2685821657736338717ULL;
    return static_cast<std::int32_t>(value % vocabulary);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: ninfer_qwen4_prefill_bench ARTIFACT WIDTH [REPEATS]\n";
        return 2;
    }
    try {
        const std::filesystem::path artifact(argv[1]);
        const std::int32_t width = parse_width(argv[2]);
        const std::int32_t repeats = argc == 4 ? parse_repeats(argv[3]) : 1;
        if (!std::filesystem::is_regular_file(artifact)) {
            throw std::invalid_argument("artifact is not a regular file");
        }
        std::vector<std::int32_t> tokens(static_cast<std::size_t>(width));
        for (std::int32_t token = 0; token < width; ++token) {
            tokens[static_cast<std::size_t>(token)] = sequence_token(token);
        }

        ninfer::DeviceContext device(0);
        std::unique_ptr<verifier::LoadedModel> model =
            verifier::LoadedModel::load(artifact, device);
        verifier::Program program(*model, device, verifier::DiagnosticSnapshots::Disabled);
        std::size_t free_before = 0;
        std::size_t total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
        for (std::int32_t run = 0; run < repeats; ++run) {
            program.reset();
            const auto begin = std::chrono::steady_clock::now();
            const auto result = program.prefill_chunk(tokens);
            const double prefill_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            if (result.begin_index != 0 || result.end_index != width - 1 ||
                program.frontier() != width) {
                throw std::runtime_error("prefill interval/frontier mismatch");
            }

            double decode_seconds = 0.0;
            if (width < verifier::kQsaCapacity) {
                const auto decode_begin = std::chrono::steady_clock::now();
                (void)program.execute_token(sequence_token(width), sequence_token(width + 1));
                decode_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_begin)
                        .count();
            }
            std::size_t free_after = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
            std::cout << "QWEN4_PREFILL width=" << width
                      << " run=" << run
                      << " page_cache=" << (run == 0 ? "ambient" : "process_warm")
                      << " final_head=included"
                      << " seconds=" << prefill_seconds
                      << " input_tok_s=" << static_cast<double>(width) / prefill_seconds
                      << " decode_seconds=" << decode_seconds
                      << " decode_tok_s=" << (decode_seconds > 0.0 ? 1.0 / decode_seconds : 0.0)
                      << " free_vram_before=" << free_before
                      << " free_vram_after=" << free_after << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen4 prefill benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
