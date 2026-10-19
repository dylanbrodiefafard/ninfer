#pragma once

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

struct KvGpuPoolSnapshot {
    std::uint32_t page_group_count = 0;
    std::uint32_t entitled_pages   = 0;
    std::uint32_t mapped_pages     = 0;
    std::uint32_t free_pages       = 0;
};

// Host-side pool counters. main is the text KV pool. spec is the MTP or DFlash
// pool and stays zero when that backend is not loaded.
struct KvGpuSnapshot {
    KvGpuPoolSnapshot main;
    KvGpuPoolSnapshot spec;
};

} // namespace ninfer::targets::qwen3_6::detail
