#include "artifact/materializer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace ninfer::artifact {
namespace {

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

// One object payload copied out of a streamed file span, to the device arena or to the pinned
// mapped-host backing.
struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
    bool host                  = false;
};

} // namespace

struct PinnedReadBuffers::EventDestroyer {
    void operator()(cudaEvent_t event) const noexcept { (void)cudaEventDestroy(event); }
};

PinnedReadBuffers::PinnedReadBuffers() {
    buffers_.reserve(Reader::stream_buffer_count);
    events_.reserve(Reader::stream_buffer_count);
    for (std::size_t i = 0; i < Reader::stream_buffer_count; ++i) {
        buffers_.emplace_back(Reader::stream_buffer_bytes, Reader::direct_io_alignment);
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        events_.emplace_back(event);
    }
}

PinnedReadBuffers::~PinnedReadBuffers() = default;

Reader::StreamBuffers PinnedReadBuffers::views() const {
    Reader::StreamBuffers result;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = std::span(static_cast<std::byte*>(buffers_[i].data()), buffers_[i].size());
    }
    return result;
}

Reader::Completion PinnedReadBuffers::release_after_copies(std::size_t slot, cudaStream_t stream) {
    cudaEvent_t event = events_.at(slot).get();
    CUDA_CHECK(cudaEventRecord(event, stream));
    return [event] { CUDA_CHECK(cudaEventSynchronize(event)); };
}

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    if (!device_arena_) { throw ArtifactError("artifact has no device tensor backing"); }
    return *device_arena_;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress) {
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.tensor_count          = plan.device_objects.size() + plan.mapped_host_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    std::vector<CopyRange> ranges;
    ranges.reserve(plan.device_objects.size() + plan.mapped_host_objects.size());
    std::uint64_t total = 0;
    if (plan.mapped_host_capacity_bytes != 0) {
        if (plan.mapped_host_capacity_bytes > static_cast<std::uint64_t>(SIZE_MAX)) {
            throw ArtifactError("artifact mapped host backing size is invalid");
        }
        out.mapped_host_ = std::make_unique<PinnedHostBuffer>(
            static_cast<std::size_t>(plan.mapped_host_capacity_bytes), 4096);
        auto* base = static_cast<std::byte*>(out.mapped_host_->data());
        for (const DeviceMaterialization& placement : plan.mapped_host_objects) {
            const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
            if (payload.data.size() != placement.bytes ||
                placement.offset > out.mapped_host_->size() - placement.bytes ||
                placement.offset % placement.alignment != 0) {
                throw ArtifactError(
                    "mapped host materialization plan does not match artifact payload");
            }
            // Pinned host memory is device-addressable at the same pointer under unified
            // addressing.
            out.objects_.at(placement.object.index).device = base + placement.offset;
            out.stats_.mapped_host_bytes += placement.bytes;
            ranges.push_back(CopyRange{
                .source_begin = payload.absolute_offset,
                .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                            "artifact tensor source range overflows u64"),
                .destination  = base + placement.offset,
                .host         = true,
            });
            total = checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
        }
    }

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.resize(payload.data.size());
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, reader.read_bytes(payload.absolute_offset, resource),
                        "artifact read bytes overflow u64");
        out.stats_.retained_resource_bytes += resource.size();
    }

    for (const DeviceMaterialization& placement : plan.device_objects) {
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        DeviceSpan storage =
            out.device_arena_->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                           static_cast<std::size_t>(placement.alignment));
        const auto actual_offset =
            static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                       static_cast<std::byte*>(out.device_arena_->base()));
        if (actual_offset != placement.offset || payload.data.size() != placement.bytes) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device = storage.data;
        ranges.push_back(CopyRange{
            .source_begin = payload.absolute_offset,
            .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                        "artifact tensor source range overflows u64"),
            .destination  = static_cast<std::byte*>(storage.data),
        });
        total = checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
    }
    if (plan.device_objects.empty()) {
        throw ArtifactError("materialization plan has no device tensors");
    }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    // Ranges closer than one direct-I/O block share a streamed span; larger gaps are skipped.
    PinnedReadBuffers staging;
    out.stats_.peak_staging_bytes = Reader::stream_buffer_count * Reader::stream_buffer_bytes;
    std::uint64_t copied          = 0;
    std::uint64_t last_published  = 0;
    const auto start              = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (std::size_t first = 0; first < ranges.size();) {
        std::size_t last = first + 1;
        while (last < ranges.size() && ranges[last].source_begin - ranges[last - 1].source_end <
                                           Reader::direct_io_alignment) {
            ++last;
        }
        const auto span_begin = ranges[first].source_begin;
        const auto span_end   = ranges[last - 1].source_end;
        std::size_t next      = first;
        const auto consume    = [&](PayloadSpan chunk, std::size_t slot) -> Reader::Completion {
            const auto chunk_end = chunk.absolute_offset + chunk.data.size();
            bool device_copies   = false;
            try {
                while (next < last && ranges[next].source_end <= chunk.absolute_offset) { ++next; }
                for (std::size_t i = next; i < last && ranges[i].source_begin < chunk_end; ++i) {
                    const CopyRange& range = ranges[i];
                    const auto begin       = std::max(chunk.absolute_offset, range.source_begin);
                    const auto end         = std::min(chunk_end, range.source_end);
                    const auto amount      = static_cast<std::size_t>(end - begin);
                    std::byte* destination =
                        range.destination + static_cast<std::size_t>(begin - range.source_begin);
                    const std::byte* source =
                        chunk.data.data() + static_cast<std::size_t>(begin - chunk.absolute_offset);
                    if (range.host) {
                        std::memcpy(destination, source, amount);
                    } else {
                        CUDA_CHECK(cudaMemcpyAsync(destination, source, amount,
                                                   cudaMemcpyHostToDevice, device.load_stream));
                        device_copies = true;
                    }
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                }
                Reader::Completion completion;
                if (device_copies) {
                    completion = staging.release_after_copies(slot, device.load_stream);
                }
                if (progress != nullptr && progress->callback && copied != last_published &&
                    copied < total) {
                    last_published = copied;
                    progress->callback("weights", copied, total);
                }
                return completion;
            } catch (...) {
                // A consumer that throws has no completion: it must settle its own copies.
                (void)cudaStreamSynchronize(device.load_stream);
                throw;
            }
        };
        out.stats_.file_bytes = checked_add(
            out.stats_.file_bytes,
            reader.stream({span_begin, span_end - span_begin}, staging.views(), consume),
            "artifact read bytes overflow u64");
        first = last;
    }
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    if (copied != total) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = copied - out.stats_.mapped_host_bytes;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

} // namespace ninfer::artifact
