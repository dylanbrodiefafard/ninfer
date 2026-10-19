#include "artifact/reader.h"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kV1Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{1},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kPayloadAlignment = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status{};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
    }

    ~MappedFile() {
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
    }

private:
    int fd_                = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

// Applies accessor to the active descriptor through std::get_if, so the noexcept accessors below
// carry no std::bad_variant_access path. A descriptor is never valueless: descriptors are built
// whole and never reassigned in place, so the terminate branch marks a broken invariant.
template <class Accessor>
decltype(auto) access_descriptor(const ObjectDescriptor& object, Accessor accessor) noexcept {
    if (const auto* tensor = std::get_if<TensorDescriptor>(&object)) { return accessor(*tensor); }
    const auto* resource = std::get_if<ResourceDescriptor>(&object);
    if (resource == nullptr) { std::terminate(); }
    return accessor(*resource);
}

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return access_descriptor(
        object, [](const auto& descriptor) -> std::string_view { return descriptor.name; });
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return access_descriptor(
        object, [](const auto& descriptor) -> std::uint64_t { return descriptor.offset; });
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return access_descriptor(
        object, [](const auto& descriptor) -> std::uint64_t { return descriptor.bytes; });
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the v2 prefix");
        }
        if (std::equal(kV1Magic.begin(), kV1Magic.end(), file.data())) {
            throw ArtifactError("NInfer artifact v1 is no longer supported; migrate it with: "
                                "python3 -m tools.artifact.migrate_v1_to_v2 <artifact>");
        }
        if (!std::equal(kMagic.begin(), kMagic.end(), file.data())) {
            throw ArtifactError("artifact magic is not NInfer v2");
        }

        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            cursor = end;
        }
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

std::string
Reader::tensor_content_identity(std::span<const TensorContentSelection> selections) const {
    struct ByteRange {
        std::uint64_t offset;
        std::uint64_t bytes;
    };

    struct SelectedTensor {
        const TensorDescriptor* descriptor;
        std::vector<ByteRange> ranges;
    };

    std::map<std::string_view, SelectedTensor> tensors;
    for (const auto& selection : selections) {
        const auto* object = find(selection.name);
        const auto* tensor = object == nullptr ? nullptr : std::get_if<TensorDescriptor>(object);
        if (tensor == nullptr) {
            throw ArtifactError("content selection is not a tensor: " +
                                std::string(selection.name));
        }
        auto [entry, inserted] = tensors.try_emplace(tensor->name, SelectedTensor{tensor, {}});
        (void)inserted;
        auto& ranges = entry->second.ranges;
        if (!selection.rows) {
            ranges.push_back({0, tensor->bytes});
            continue;
        }
        const auto [first, rows] = *selection.rows;
        if (tensor->shape.size() != 2 || rows == 0 || first >= tensor->shape[0] ||
            rows > tensor->shape[0] - first) {
            throw ArtifactError("content row selection is out of range: " + tensor->name);
        }
        const auto columns = tensor->shape[1];
        switch (tensor->layout) {
        case StorageLayout::ContiguousLeV1: {
            const auto stride = tensor->bytes / tensor->shape[0];
            ranges.push_back({first * stride, rows * stride});
            break;
        }
        case StorageLayout::RowSplitK128V1: {
            const auto geometry     = row_split_geometry(tensor->format, tensor->shape);
            const auto low_stride   = geometry.groups_per_row * geometry.low_bytes_per_group;
            const auto high_stride  = geometry.groups_per_row * geometry.high_bytes_per_group;
            const auto scale_stride = geometry.groups_per_row * 2;
            ranges.push_back({first * low_stride, rows * low_stride});
            if (high_stride != 0) {
                ranges.push_back(
                    {geometry.high_plane_offset + first * high_stride, rows * high_stride});
            }
            ranges.push_back(
                {geometry.scale_plane_offset + first * scale_stride, rows * scale_stride});
            break;
        }
        case StorageLayout::BlockScaleK16M128x4V1: {
            if (first % 128 != 0 || rows % 128 != 0) {
                throw ArtifactError("NVFP4 content rows must align to 128-row tiles: " +
                                    tensor->name);
            }
            const auto geometry = block_scale_geometry(tensor->format, tensor->shape);
            ranges.push_back({first * (columns / 2), rows * (columns / 2)});
            // Scale swizzling is contained within each 128-row tile.
            ranges.push_back(
                {geometry.scale_plane_offset + first * (columns / 16), rows * (columns / 16)});
            ranges.push_back({geometry.weight_divisor_offset, 4});
            break;
        }
        case StorageLayout::RowScaleV1: {
            const auto geometry = row_scale_geometry(tensor->format, tensor->shape);
            ranges.push_back({first * columns, rows * columns});
            ranges.push_back({geometry.scale_plane_offset + first * 2, rows * 2});
            break;
        }
        }
    }
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                          EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw ArtifactError("cannot initialize artifact tensor SHA-256");
    }
    const auto update = [&](const void* data, std::size_t bytes) {
        if (EVP_DigestUpdate(context.get(), data, bytes) != 1) {
            throw ArtifactError("cannot update artifact tensor SHA-256");
        }
    };
    for (auto& [name, selected] : tensors) {
        auto& ranges = selected.ranges;
        std::sort(ranges.begin(), ranges.end(),
                  [](const auto& left, const auto& right) { return left.offset < right.offset; });
        std::vector<ByteRange> merged;
        for (const auto range : ranges) {
            if (!merged.empty() && range.offset <= merged.back().offset + merged.back().bytes) {
                auto& previous = merged.back();
                previous.bytes =
                    std::max(previous.bytes, range.offset + range.bytes - previous.offset);
            } else {
                merged.push_back(range);
            }
        }
        Json encoded_ranges = Json::array();
        for (const auto range : merged) { encoded_ranges.push_back({range.offset, range.bytes}); }
        const auto& tensor           = *selected.descriptor;
        const std::string descriptor = Json::array({name, tensor.shape, format_name(tensor.format),
                                                    layout_name(tensor.layout), encoded_ranges})
                                           .dump();
        update(descriptor.data(), descriptor.size());
        const auto data = payload(name).data;
        for (const auto range : merged) {
            update(data.data() + range.offset, static_cast<std::size_t>(range.bytes));
        }
    }
    std::array<unsigned char, 32> digest{};
    unsigned int bytes = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &bytes) != 1 || bytes != digest.size()) {
        throw ArtifactError("cannot finalize artifact tensor SHA-256");
    }
    constexpr std::string_view hex = "0123456789abcdef";
    std::string result             = "tensor-sha256-v1:";
    for (const auto byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 15]);
    }
    return result;
}

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
