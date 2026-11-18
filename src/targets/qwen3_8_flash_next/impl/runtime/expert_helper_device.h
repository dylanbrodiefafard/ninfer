#pragma once

// Narrow HIP graph owner used only by the optional Flash-Next helper. Runtime headers remain
// private to expert_helper.hip, never Program/Forward or ordinary CUDA Ops.
#include "core/expert_mailbox.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next {
[[nodiscard]] std::string_view expert_helper_toolchain() noexcept;

struct ExpertHelperBank {
    const std::uint64_t* records = nullptr;
    const float* matrix_scales   = nullptr;
    std::array<std::uint32_t, 3> f3_books{};
};

class ExpertHelperExecution {
public:
    ExpertHelperExecution(ExpertMailboxView mailbox, const std::array<ExpertHelperBank, 48>& banks,
                          std::int32_t lanes, bool speculative, const HipStream& stream);
    ~ExpertHelperExecution();
    ExpertHelperExecution(const ExpertHelperExecution&)            = delete;
    ExpertHelperExecution& operator=(const ExpertHelperExecution&) = delete;
    void launch(std::uint32_t tokens);
    // Joins the completion event on the constructor's fixed stream; no CPU polling tax.
    void join();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::targets::qwen3_8_flash_next
