#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::size_t kv_ram_capacity_bytes = 0;
    std::size_t kv_disk_capacity_bytes = 0;
    std::filesystem::path kv_disk_location;
    KvDiskCompress kv_disk_compress = KvDiskCompress::Off;
    std::uint32_t prefill_chunk  = 4096;
    int device                   = 0;

    KvCacheStorage kv_cache = KvCacheStorage::Nvfp4;
    bool sage_attn = false;
    float keep_frac = 1.0f;
    float xattn_tau = 1.0f;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    bool use_cuda_graph = true;

    bool capture_context_checkpoint = false;
    std::optional<std::vector<std::uint32_t>> context_checkpoint_marks;

    bool raw_output      = false;
    bool print_token_ids = false;
    bool enable_thinking = true;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy = false;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
