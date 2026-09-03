#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

struct KvDiskCopySeconds {
    double save = 0;
    double load = 0;
    double h2d  = 0;
};

struct KvDiskSnapshot {
    std::size_t capacity_bytes = 0;
    std::size_t used_bytes     = 0;
    std::size_t entry_count    = 0;
    std::uint64_t captures     = 0;
    std::uint64_t restores     = 0;
    std::uint64_t evictions    = 0;
    std::uint64_t drops        = 0;
    double save_seconds        = 0;
    double load_seconds        = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
