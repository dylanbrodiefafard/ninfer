#pragma once

// Minimal NumPy .npy reading and writing for the Qwen3.8-Flash-Next parity and timing driver:
// C-order arrays of one exact little-endian dtype, format versions 1 and 2.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::tools::flash_next {

struct NpyArray {
    std::vector<std::int64_t> shape;
    std::vector<std::byte> data;
};

/**
 * Reads a C-order .npy whose dtype descriptor is exactly `descr` (for example "<i4"). Throws
 * std::runtime_error on an I/O failure, another dtype, Fortran order, a malformed header, or a
 * payload whose size differs from the shape.
 */
[[nodiscard]] NpyArray read_npy(const std::filesystem::path& path, std::string_view descr,
                                std::size_t element_bytes);

/** A version 1.0 .npy file of a known shape and dtype, written in C order by appending rows. */
class NpyWriter {
public:
    NpyWriter(const std::filesystem::path& path, std::string_view descr,
              std::span<const std::int64_t> shape);

    void write(std::span<const std::byte> bytes);

private:
    std::ofstream file_;
    std::filesystem::path path_;
};

} // namespace ninfer::tools::flash_next
