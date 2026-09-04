#include "artifact/materializer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 8;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

class Slot {
public:
    explicit Slot(std::size_t bytes)
        : buffer(bytes + Reader::direct_io_alignment - 1) {
        const auto address = reinterpret_cast<std::uintptr_t>(buffer.data());
        const auto aligned =
            (address + Reader::direct_io_alignment - 1) / Reader::direct_io_alignment *
            Reader::direct_io_alignment;
        data_ = reinterpret_cast<std::byte*>(aligned);
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            CUDA_CHECK(cudaEventSynchronize(event));
            pending = false;
        }
    }

    [[nodiscard]] std::byte* data() const noexcept { return data_; }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;

private:
    std::byte* data_ = nullptr;
};

class DirectReadPool {
public:
    explicit DirectReadPool(std::size_t worker_count) {
        workers_.reserve(worker_count);
        try {
            for (std::size_t i = 0; i < worker_count; ++i) {
                workers_.emplace_back([this] { worker_loop(); });
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
            for (std::thread& worker : workers_) { worker.join(); }
            throw;
        }
    }

    ~DirectReadPool() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread& worker : workers_) { worker.join(); }
    }

    DirectReadPool(const DirectReadPool&)            = delete;
    DirectReadPool& operator=(const DirectReadPool&) = delete;

    std::future<std::size_t> submit(const Reader& reader, std::uint64_t source,
                                    std::span<std::byte> destination,
                                    cudaEvent_t reusable_after = nullptr) {
        Task task{
            .reader         = &reader,
            .source         = source,
            .destination    = destination,
            .reusable_after = reusable_after,
        };
        std::future<std::size_t> result = task.result.get_future();
        {
            std::lock_guard lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        ready_.notify_one();
        return result;
    }

private:
    struct Task {
        const Reader* reader = nullptr;
        std::uint64_t source = 0;
        std::span<std::byte> destination;
        cudaEvent_t reusable_after = nullptr;
        std::promise<std::size_t> result;
    };

    void worker_loop() {
        while (true) {
            Task task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (tasks_.empty()) { return; }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }

            try {
                if (task.reusable_after != nullptr) {
                    CUDA_CHECK(cudaEventSynchronize(task.reusable_after));
                }
                task.result.set_value(task.reader->read_direct(task.source, task.destination));
            } catch (...) {
                task.result.set_exception(std::current_exception());
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Task> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

struct ReadChunk {
    std::uint64_t source   = 0;
    std::size_t request    = 0;
    std::uint64_t required = 0;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte>
MaterializedArtifact::mapped_tensor_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].mapped_tensor.empty()) {
        throw ArtifactError("object handle does not name a mapped host tensor");
    }
    return objects_[handle.index].mapped_tensor;
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

std::shared_ptr<const void>
MaterializedArtifact::retain_reader_mapping(const Reader& reader) {
    return reader.retain_mapping();
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
    if (capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    if (capacity != 0) {
        out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
    }
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.mapped_tensor_count   = plan.mapped_tensor_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    if (!plan.mapped_tensor_objects.empty()) {
        out.mapped_backing_ = MaterializedArtifact::retain_reader_mapping(reader);
        for (const MappedTensorMaterialization& placement : plan.mapped_tensor_objects) {
            const PayloadSpan payload =
                reader.payload(reader.objects().at(placement.object.index));
            if (payload.data.size() != placement.bytes) {
                throw ArtifactError("materialization plan does not match mapped tensor payload");
            }
            out.objects_.at(placement.object.index).mapped_tensor = payload.data;
            out.stats_.mapped_tensor_bytes =
                checked_add(out.stats_.mapped_tensor_bytes, placement.bytes,
                            "mapped tensor byte count overflows u64");
        }
    }

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::vector<CopyRange> ranges;
    ranges.reserve(plan.device_objects.size());
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;
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
    if (ranges.empty()) { return out; }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                   "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t chunk_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    std::vector<ReadChunk> chunks;
    chunks.reserve(static_cast<std::size_t>(
        1 + (aligned_read_bytes - 1) / static_cast<std::uint64_t>(chunk_bytes)));
    std::size_t largest_request = 0;
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += chunk_bytes) {
            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                chunk_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            chunks.push_back(ReadChunk{
                .source   = source,
                .request  = request,
                .required = std::min<std::uint64_t>(request, remaining),
            });
            largest_request = std::max(largest_request, request);
        }
    }
    const std::size_t slot_count = std::min(kMaximumSlotCount, chunks.size());
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(largest_request));
    }
    out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(largest_request) * slot_count;

    DirectReadPool read_pool(slot_count);
    std::vector<std::future<std::size_t>> reads(slot_count);
    const auto submit_read = [&](std::size_t chunk_index, cudaEvent_t reusable_after = nullptr) {
        Slot& slot             = *slots[chunk_index % slot_count];
        const ReadChunk& chunk = chunks[chunk_index];
        auto destination       = std::span<std::byte>(slot.data(), chunk.request);
        reads[chunk_index % slot_count] =
            read_pool.submit(reader, chunk.source, destination, reusable_after);
    };
    std::size_t next_range = 0;
    const auto start       = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (std::size_t i = 0; i < slot_count; ++i) { submit_read(i); }

    for (std::size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const ReadChunk& chunk = chunks[chunk_index];
        Slot& slot             = *slots[chunk_index % slot_count];
        const std::size_t bytes_read = reads[chunk_index % slot_count].get();
        if (bytes_read < chunk.required) {
            throw ArtifactError("direct artifact read ended before the planned tensor range");
        }
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
        const std::uint64_t chunk_end =
            checked_add(chunk.source, bytes_read, "artifact direct I/O result overflows u64");

        while (next_range < ranges.size() && ranges[next_range].source_end <= chunk.source) {
            ++next_range;
        }
        std::size_t range_index = next_range;
        while (range_index < ranges.size() && ranges[range_index].source_begin < chunk_end) {
            const CopyRange& range         = ranges[range_index];
            const std::uint64_t copy_begin = std::max(chunk.source, range.source_begin);
            const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
            if (copy_begin < copy_end) {
                const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                CUDA_CHECK(cudaMemcpyAsync(
                    range.destination + static_cast<std::size_t>(copy_begin - range.source_begin),
                    slot.data() + static_cast<std::size_t>(copy_begin - chunk.source),
                    amount, cudaMemcpyHostToDevice, device.load_stream));
                copied = checked_add(copied, amount, "artifact copied byte count overflows u64");
            }
            if (range.source_end <= chunk_end) {
                ++range_index;
            } else {
                break;
            }
        }
        next_range = range_index;
        CUDA_CHECK(cudaEventRecord(slot.event, device.load_stream));
        slot.pending = true;

        const std::size_t next_chunk = chunk_index + slot_count;
        if (next_chunk < chunks.size()) { submit_read(next_chunk, slot.event); }

        if (progress != nullptr && progress->callback && copied != last_published &&
            copied < total) {
            last_published = copied;
            progress->callback("weights", copied, total);
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    CUDA_CHECK(cudaStreamSynchronize(device.load_stream));
    if (copied != total || next_range != ranges.size()) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

} // namespace ninfer::artifact
