#include "npy.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::tools::flash_next {
namespace {

constexpr std::string_view kMagic = "\x93NUMPY";

std::vector<std::int64_t> parse_shape(const std::string& header, const std::string& label) {
    const std::size_t at = header.find("'shape': (");
    if (at == std::string::npos) { throw std::runtime_error(label + " has no shape"); }
    const std::size_t open  = at + 10;
    const std::size_t close = header.find(')', open);
    if (close == std::string::npos) { throw std::runtime_error(label + " has a malformed shape"); }
    std::vector<std::int64_t> shape;
    for (std::size_t cursor = open; cursor < close;) {
        const std::size_t end   = std::min(header.find(',', cursor), close);
        const std::string field = header.substr(cursor, end - cursor);
        if (field.find_first_not_of(' ') != std::string::npos) {
            std::size_t used          = 0;
            const long long dimension = std::stoll(field, &used);
            if (dimension < 0 || field.find_first_not_of(' ', used) != std::string::npos) {
                throw std::runtime_error(label + " has a malformed shape");
            }
            shape.push_back(dimension);
        }
        cursor = end + 1;
    }
    return shape;
}

} // namespace

NpyArray read_npy(const std::filesystem::path& path, std::string_view descr,
                  std::size_t element_bytes) {
    const std::string label = path.string();
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::runtime_error("cannot open " + label); }
    const auto file_bytes = static_cast<std::size_t>(file.tellg());
    file.seekg(0);
    std::array<char, 10> preamble{};
    file.read(preamble.data(), preamble.size());
    if (!file || std::string_view(preamble.data(), kMagic.size()) != kMagic) {
        throw std::runtime_error(label + " is not a .npy file");
    }
    const auto major = static_cast<unsigned char>(preamble[6]);
    std::size_t header_bytes =
        static_cast<unsigned char>(preamble[8]) |
        (static_cast<std::size_t>(static_cast<unsigned char>(preamble[9])) << 8U);
    std::size_t prefix = preamble.size();
    if (major >= 2) {
        std::array<char, 2> high{};
        file.read(high.data(), high.size());
        header_bytes |= (static_cast<std::size_t>(static_cast<unsigned char>(high[0])) << 16U) |
                        (static_cast<std::size_t>(static_cast<unsigned char>(high[1])) << 24U);
        prefix += high.size();
    }
    std::string header(header_bytes, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header_bytes));
    if (!file || header.find("'descr': '" + std::string(descr) + "'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error(label + " must be a C-order '" + std::string(descr) + "' array");
    }
    NpyArray out{.shape = parse_shape(header, label), .data = {}};
    std::size_t elements = 1;
    for (const std::int64_t dimension : out.shape) {
        elements *= static_cast<std::size_t>(dimension);
    }
    if (prefix + header_bytes > file_bytes ||
        file_bytes - prefix - header_bytes != elements * element_bytes) {
        throw std::runtime_error(label + ": the payload size differs from its shape");
    }
    out.data.resize(elements * element_bytes);
    file.read(reinterpret_cast<char*>(out.data.data()),
              static_cast<std::streamsize>(out.data.size()));
    if (!file) { throw std::runtime_error("cannot read " + label); }
    return out;
}

NpyWriter::NpyWriter(const std::filesystem::path& path, std::string_view descr,
                     std::span<const std::int64_t> shape)
    : file_(path, std::ios::binary | std::ios::trunc), path_(path) {
    std::string dims;
    for (const std::int64_t dimension : shape) { dims += std::to_string(dimension) + ", "; }
    std::string header = "{'descr': '" + std::string(descr) +
                         "', 'fortran_order': False, 'shape': (" + dims + "), }";
    // Version 1.0: a 10-byte preamble, then the header padded with spaces so that the payload
    // starts at a 64-byte multiple, ending in a newline.
    const std::size_t total = (kMagic.size() + 4 + header.size() + 1 + 63) / 64 * 64;
    header.append(total - kMagic.size() - 4 - header.size() - 1, ' ');
    header.push_back('\n');
    const std::size_t header_bytes = header.size();
    file_.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    const std::array<char, 4> version_and_length = {1, 0, static_cast<char>(header_bytes & 0xffU),
                                                    static_cast<char>(header_bytes >> 8U)};
    file_.write(version_and_length.data(), version_and_length.size());
    file_.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!file_) { throw std::runtime_error("cannot write " + path.string()); }
}

void NpyWriter::write(std::span<const std::byte> bytes) {
    file_.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!file_) { throw std::runtime_error("cannot write " + path_.string()); }
}

} // namespace ninfer::tools::flash_next
