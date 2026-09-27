#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

enum class TensorPlacement : std::uint8_t {
    Device,
    // Pinned host memory read directly by kernels through its unified device address.
    MappedHost,
    ValidateOnly,
};

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count                 = 0;
    std::uint64_t device_capacity_bytes      = 0;
    std::uint64_t mapped_host_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    // Offsets are into the one pinned host backing; tensors keep their device alignment.
    std::vector<DeviceMaterialization> mapped_host_objects;
    std::vector<HostMaterialization> host_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    [[nodiscard]] bool contains(std::string_view name) const noexcept;
    [[nodiscard]] const ObjectDescriptor* find(std::string_view name) const noexcept;
    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    void materialize_on_mapped_host(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);
    void plan_tensor(ObjectHandle handle, std::uint64_t& capacity_bytes,
                     std::vector<DeviceMaterialization>& objects);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
