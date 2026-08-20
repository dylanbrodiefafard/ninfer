#pragma once

#include <cstdint>

namespace ninfer::targets::qwen3_6 {

enum class DFlashKind : std::uint8_t { None = 0, V1 = 1, DFlash2 = 2 };

} // namespace ninfer::targets::qwen3_6
