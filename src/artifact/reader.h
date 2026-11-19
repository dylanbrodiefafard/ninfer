#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericFormat {
    BF16,
    FP32,
    I32,
    Q4G64_F16S,
    Q5G64_F16S,
    Q6G64_F16S,
    W8G32_F16S,
    NVFP4,
    FP8_E4M3FN_ROW_BF16S,
    F3G32_E4M3FN,
    E2M1G16_E4M3FN,
};

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
    MoeExpertTilesV1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;
std::string_view encoding_name(ResourceEncoding encoding) noexcept;

std::uint64_t tensor_alignment(StorageLayout layout) noexcept;
std::uint64_t resource_alignment(ResourceEncoding encoding) noexcept;
std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape);

struct RowSplitGeometry {
    std::uint64_t rows                 = 0;
    std::uint64_t columns              = 0;
    std::uint64_t padded_columns       = 0;
    std::uint64_t group_size           = 0;
    std::uint64_t groups_per_row       = 0;
    std::uint64_t low_bytes_per_group  = 0;
    std::uint64_t high_bytes_per_group = 0;
    std::uint64_t low_plane_bytes      = 0;
    std::uint64_t high_plane_offset    = 0;
    std::uint64_t high_plane_bytes     = 0;
    std::uint64_t scale_plane_offset   = 0;
    std::uint64_t scale_plane_bytes    = 0;
    std::uint64_t encoded_bytes        = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowScaleGeometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

// moe-expert-tiles-v1 over [experts, hidden, intermediate]: per expert, gate|up chunks of 16 gate
// and 16 up rows by 128 K, then down chunks of 32 rows by 128 K, padded to 256 bytes.
struct ExpertTilesGeometry {
    std::uint64_t experts        = 0;
    std::uint64_t hidden         = 0;
    std::uint64_t intermediate   = 0;
    std::uint64_t chunk_bytes    = 0;
    std::uint64_t gate_up_chunks = 0;
    std::uint64_t down_chunks    = 0;
    std::uint64_t record_bytes   = 0;
    std::uint64_t encoded_bytes  = 0;
};

ExpertTilesGeometry expert_tiles_geometry(NumericFormat format,
                                          std::span<const std::uint64_t> shape);

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset;
    std::uint64_t bytes;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

std::string_view object_name(const ObjectDescriptor& object) noexcept;
std::uint64_t object_offset(const ObjectDescriptor& object) noexcept;
std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept;

struct PayloadSpan {
    std::uint64_t absolute_offset;
    std::span<const std::byte> data;
};

struct ArtifactIdentity {
    std::string model_id;
    std::string weights_id;

    bool operator==(const ArtifactIdentity&) const = default;
};

struct TensorRowRange {
    std::uint64_t first_row;
    std::uint64_t rows;
};

// The name borrows from the Reader's inventory. An absent row range selects the whole tensor;
// row ranges select a rank-two matrix, including its corresponding codes, scales, and divisor.
struct TensorContentSelection {
    std::string_view name;
    std::optional<TensorRowRange> rows;
};

struct FileRange {
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

class Reader {
public:
    static constexpr std::size_t direct_io_alignment = 4096;
    static constexpr std::size_t stream_buffer_bytes = std::size_t{64} << 20;
    static constexpr std::size_t stream_buffer_count = 4;
    using StreamBuffers = std::array<std::span<std::byte>, stream_buffer_count>;
    using Completion    = std::function<void()>;
    using Consumer      = std::function<Completion(PayloadSpan, std::size_t)>;

    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    const ArtifactIdentity& identity() const noexcept;
    // SHA-256 of selected tensor descriptors and encoded bytes, canonicalized by name/range.
    // Selection belongs to the caller's numerical/state contract. File metadata, resources,
    // object placement, and unselected bytes are excluded. Partial NVFP4 rows must align to
    // 128-row scale tiles. Throws ArtifactError for invalid selections or digest failures.
    // Reads selected bytes; the artifact must remain immutable during use.
    [[nodiscard]] std::string
    tensor_content_identity(std::span<const TensorContentSelection> selections) const;
    const std::vector<ObjectDescriptor>& objects() const noexcept;
    const ObjectDescriptor* find(std::string_view name) const noexcept;

    std::uint64_t file_bytes() const noexcept;
    std::uint64_t payload_offset() const noexcept;
    PayloadSpan payload(const ObjectDescriptor& object) const;
    PayloadSpan payload(std::string_view name) const;
    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const;
    // Direct I/O with exact unaligned edges for metadata, resources and hashing windows; no
    // page-cache reads. Returns the physical bytes read, including edge padding.
    [[nodiscard]] std::uint64_t read_bytes(std::uint64_t absolute_offset,
                                           std::span<std::byte> destination) const;
    // Sequential O_DIRECT reads through four caller-owned, disjoint, 4096-aligned buffers of one
    // equal size, a multiple of 4096 (stream_buffer_bytes in production; pinned for device
    // copies). The consumer sees only the requested bytes, in order.
    // Its optional completion must settle all accesses even when it throws. Before reuse and
    // on every exit, completions are joined; a throwing consumer must itself settle accesses
    // it has not returned a completion for. Buffers and captured completion owners must outlive
    // this call. unit_bytes keeps each delivered chunk a whole number of logical records;
    // the range must contain whole units. Returns physical bytes read, including edge padding.
    // Read, range, alignment and short-read failures throw ArtifactError.
    [[nodiscard]] std::uint64_t stream(FileRange range, const StreamBuffers& buffers,
                                       const Consumer& consume, std::size_t unit_bytes = 1) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
