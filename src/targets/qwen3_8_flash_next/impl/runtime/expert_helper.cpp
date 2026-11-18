#include "runtime/expert_helper.h"

#include "core/arena.h"
#include "core/device.h"
#include "load/binder.h"
#include "runtime/expert_helper_device.h"
#include "runtime/expert_helper_profile.h"
#include "runtime/expert_helper_upload.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {
std::uint32_t capacity(std::int32_t lanes, bool speculative) {
    if (lanes < 1 || lanes > 12) {
        throw std::invalid_argument("expert helper lanes must be in [1,12]");
    }
    return static_cast<std::uint32_t>(lanes * (speculative ? 4 : 1));
}

std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

std::vector<ops::MoeExpertCodec> codec_map(const ExpertHostStore& store) {
    if (store.layers() != 48) {
        throw std::invalid_argument("expert helper needs all 48 expert layers");
    }
    std::vector<ops::MoeExpertCodec> result;
    result.reserve(48 * 512);
    for (std::int32_t l = 0; l < 48; ++l) {
        for (std::int32_t e = 0; e < 512; ++e) { result.push_back(store.codec(l, e)); }
    }
    return result;
}
} // namespace

struct ExpertHelper::Impl {
    HipDevice device;
    HipDeviceInfo initial;
    HipStream stream;
    void* const cuda_queue;
    ExpertHelperPlacement placement;
    ExpertSlotPlan primary;
    HipBuffer records, table, scales;
    DeviceArena cuda_storage;
    // Reverse destruction: graphs, mailbox registration/backing, then device banks/panels.
    ExpertMailbox mailbox;
    ExpertHelperCudaView cuda{};
    ExpertHelperMemory memory;
    std::unique_ptr<ExpertHelperExecution> execution;

    Impl(const LoadedModel& model, const ExpertSlotPlan& fixed, std::string_view pci,
         std::int32_t lanes, bool speculative, void* cuda_stream, std::size_t primary_flex_slots)
        : device(pci), initial(device.info()), cuda_queue(cuda_stream),
          placement(plan_expert_helper(codec_map(model.experts()), kExpertHelperCalibCounts,
                                       fixed.slots, initial.free_bytes)),
          // Placement never removes the protected F entries of either codec. Capping the
          // original two-pool plan by remaining counts therefore leaves its capacities intact;
          // re-splitting rounded record bytes would incorrectly discard slot-granularity slack.
          primary(fixed), records(placement.helper_bytes[0] + placement.helper_bytes[1]),
          table(48 * 512 * sizeof(std::uint64_t)), scales(48 * 512 * 3 * sizeof(float)),
          cuda_storage(ExpertHelper::cuda_device_bytes(capacity(lanes, speculative))),
          mailbox(capacity(lanes, speculative)) {
        const auto max_t = capacity(lanes, speculative);
        cuda.mailbox     = mailbox.cuda_view();
        const auto eligibility =
            cuda_storage.alloc_bytes(placement.local_eligibility.size() * sizeof(std::uint32_t));
        cuda.local_eligibility = static_cast<const std::uint32_t*>(eligibility.data);
        cuda.local_partial =
            static_cast<float*>(cuda_storage.alloc_bytes(max_t * kExpertPartialBytes).data);
        cuda.remote_partial =
            static_cast<float*>(cuda_storage.alloc_bytes(max_t * kExpertPartialBytes).data);
        cuda.compact_x = static_cast<std::uint16_t*>(
            cuda_storage.alloc_bytes(max_t * kExpertActivationBytes).data);
        cuda.compact_ids = static_cast<std::int32_t*>(
            cuda_storage.alloc_bytes(max_t * (kExpertMetadataBytes / 2)).data);
        cuda.compact_weights =
            static_cast<float*>(cuda_storage.alloc_bytes(max_t * (kExpertMetadataBytes / 2)).data);
        cuda.compact_map =
            static_cast<std::int32_t*>(cuda_storage.alloc_bytes(max_t * sizeof(std::int32_t)).data);
        cuda.compact_count =
            static_cast<std::uint32_t*>(cuda_storage.alloc_bytes(sizeof(std::uint32_t)).data);
        CUDA_CHECK(cudaMemsetAsync(cuda_storage.base(), 0, cuda_storage.capacity(),
                                   static_cast<cudaStream_t>(cuda_stream)));
        CUDA_CHECK(cudaMemcpyAsync(eligibility.data, placement.local_eligibility.data(),
                                   placement.local_eligibility.size() * sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, static_cast<cudaStream_t>(cuda_stream)));
        CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream)));

        std::vector<ExpertHelperUpload> uploads;
        std::vector<std::uint64_t> addresses(48 * 512, 0);
        std::size_t offset = 0;
        const auto base    = reinterpret_cast<std::uintptr_t>(records.data());
        if (base % 256 != 0) {
            throw HipError("expert helper record allocation is not 256-byte aligned");
        }
        for (std::size_t i = 0; i < addresses.size(); ++i) {
            if (!placement.helper_owns(i)) { continue; }
            const auto l      = static_cast<std::int32_t>(i / 512),
                       e      = static_cast<std::int32_t>(i % 512);
            const auto record = model.experts().record(l, e);
            uploads.push_back({record, offset});
            addresses[i] =
                (base + offset) | static_cast<std::uint64_t>(model.experts().codec(l, e));
            offset += record.size();
        }
        if (offset != records.size() ||
            uploads.size() != static_cast<std::size_t>(placement.helper_counts[0]) +
                                  static_cast<std::size_t>(placement.helper_counts[1])) {
            throw HipError("expert helper record inventory mismatch");
        }
        const auto upload = upload_expert_helper_records(uploads, records, stream);
        std::vector<float> matrix_scales(48 * 512 * 3);
        std::array<ExpertHelperBank, 48> banks{};
        for (std::size_t l = 0; l < 48; ++l) {
            const auto& source = model.view().layers[l].experts;
            CUDA_CHECK(cudaMemcpy(matrix_scales.data() + l * 512 * 3, source.matrix_scales.data,
                                  512 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
            banks[l] = {.records       = static_cast<const std::uint64_t*>(table.data()) + l * 512,
                        .matrix_scales = static_cast<const float*>(scales.data()) + l * 512 * 3,
                        .f3_books      = source.f3_books};
        }
        // Metadata is small, exact and kept alive through stream completion, including unwinds.
        std::vector<std::uint64_t> check_table(addresses.size());
        std::vector<float> check_scales(matrix_scales.size());
        try {
            hip_copy_to_device(table.data(), addresses.data(), table.size(), stream);
            hip_copy_to_device(scales.data(), matrix_scales.data(), scales.size(), stream);
            hip_copy_to_host(check_table.data(), table.data(), table.size(), stream);
            hip_copy_to_host(check_scales.data(), scales.data(), scales.size(), stream);
            stream.synchronize();
            if (check_table != addresses ||
                std::memcmp(check_scales.data(), matrix_scales.data(), scales.size()) != 0) {
                throw HipError("expert helper metadata upload mismatch");
            }
        } catch (...) {
            try {
                stream.synchronize();
            } catch (...) { std::terminate(); }
            throw;
        }
        qualify_expert_mailbox(mailbox, cuda_stream, stream);
        execution        = std::make_unique<ExpertHelperExecution>(mailbox.hip_view(), banks, lanes,
                                                                   speculative, stream);
        const auto after = device.info();
        const auto used =
            initial.free_bytes > after.free_bytes ? initial.free_bytes - after.free_bytes : 0;
        const auto nonrecords = used > records.size() ? used - records.size() : 0;
        if (nonrecords > kHelperReserveBytes - 2 * kHelperUploadStagingBytes) {
            throw HipError("expert helper non-record allocations exceed 1 GiB reserve");
        }
        memory = {.free_at_start_bytes    = initial.free_bytes,
                  .reserve_bytes          = kHelperReserveBytes,
                  .record_bytes           = records.size(),
                  .nonrecord_device_bytes = nonrecords,
                  .cuda_device_bytes      = cuda_storage.capacity(),
                  .mailbox_host_bytes     = mailbox.bytes(),
                  .record_counts          = placement.helper_counts};
        std::fprintf(stderr,
                     "expert-helper pci=%s name=%s arch=%s HIP=%d free=%zu reserve=%zu records=%zu "
                     "nonrecord=%zu cuda=%zu mailbox=%zu transport=T1-47:mapped,T48:DMA "
                     "upload=%.3fs checksum=ok nonce=ok\n",
                     initial.pci.c_str(), initial.name.c_str(), initial.architecture.c_str(),
                     initial.runtime_version, initial.free_bytes, kHelperReserveBytes,
                     records.size(), nonrecords, memory.cuda_device_bytes, mailbox.bytes(),
                     upload.seconds);
        std::fprintf(
            stderr,
            "expert-helper fixed=%d/%d local=%d/%d helper=%d/%d codec-bytes=%zu/%zu "
            "calib=%llu/%llu zero-flex-full=%d\n",
            fixed.slots[0], fixed.slots[1], primary.slots[0], primary.slots[1],
            placement.helper_counts[0], placement.helper_counts[1], placement.helper_bytes[0],
            placement.helper_bytes[1], static_cast<unsigned long long>(placement.helper_routes),
            static_cast<unsigned long long>(placement.total_routes), placement.full_coverage);
        const auto toolchain = expert_helper_toolchain();
        std::fprintf(stderr,
                     "expert-helper HIP-clang=%.*s primary-flex-F3=%zu remaining-codecs=%d/%d\n",
                     static_cast<int>(toolchain.size()), toolchain.data(), primary_flex_slots,
                     placement.remaining_counts[0], placement.remaining_counts[1]);
        for (std::size_t l = 0; l < 48; ++l) {
            std::fprintf(
                stderr, "expert-helper layer=%zu counts=%d/%d codec-bytes=%zu/%zu\n", l,
                placement.layer_counts[l][0], placement.layer_counts[l][1],
                static_cast<std::size_t>(placement.layer_counts[l][0]) * ops::kMoeNvfp4RecordBytes,
                static_cast<std::size_t>(placement.layer_counts[l][1]) * ops::kMoeF3RecordBytes);
        }
    }
};

ExpertHelper::ExpertHelper(const LoadedModel& model, const ExpertSlotPlan& fixed_primary,
                           std::string_view pci, std::int32_t lanes, bool speculative,
                           void* cuda_stream, std::size_t primary_flex_slots)
    : impl_(std::make_unique<Impl>(model, fixed_primary, pci, lanes, speculative, cuda_stream,
                                   primary_flex_slots)) {}

ExpertHelper::~ExpertHelper() {
    cancel_and_drain();
    // Keep banks, graph storage and both runtimes' panels together if a join failed. The
    // mailbox stops its watchdog before retention; no resources can outlive their backing.
    if (!impl_->mailbox.drained()) {
        std::fprintf(stderr, "expert helper retaining unjoined resources %p until process exit\n",
                     static_cast<void*>(impl_.release()));
    }
}

std::size_t ExpertHelper::cuda_device_bytes(std::uint32_t max_tokens) {
    if (max_tokens == 0 || max_tokens > 48) {
        throw std::invalid_argument("expert helper CUDA capacity must be [1,48]");
    }
    return 48 * 16 * sizeof(std::uint32_t) + 2 * aligned(max_tokens * kExpertPartialBytes) +
           aligned(max_tokens * kExpertActivationBytes) +
           2 * aligned(max_tokens * (kExpertMetadataBytes / 2)) +
           aligned(max_tokens * sizeof(std::int32_t)) + aligned(sizeof(std::uint32_t));
}

const ExpertSlotPlan& ExpertHelper::primary_plan() const noexcept { return impl_->primary; }

const ExpertHelperPlacement& ExpertHelper::placement() const noexcept { return impl_->placement; }

ExpertHelperCudaView ExpertHelper::cuda_view() const noexcept { return impl_->cuda; }

ExpertHelperMemory ExpertHelper::memory() const noexcept { return impl_->memory; }

std::uint64_t ExpertHelper::begin_round(std::uint32_t tokens) {
    impl_->device.bind();
    const auto sequence = impl_->mailbox.begin(tokens, impl_->cuda_queue, impl_->stream);
    try {
        impl_->execution->launch(tokens);
    } catch (...) {
        impl_->mailbox.cancel_and_drain();
        throw;
    }
    return sequence;
}

void ExpertHelper::finish_round() {
    try {
        impl_->device.bind();
        impl_->mailbox.finish();
        // The stream is already joined; retire the execution's pending event for the next round.
        impl_->execution->join();
    } catch (...) {
        impl_->mailbox.cancel_and_drain();
        throw;
    }
}

void ExpertHelper::check_failure() const {
    const auto error = impl_->mailbox.failure();
    if (error.code != ExpertWireError::None) {
        throw HipError("expert helper poisoned sequence=" + std::to_string(error.sequence) +
                       " layer=" + std::to_string(error.layer) +
                       " side=" + std::to_string(static_cast<unsigned>(error.side)) +
                       " code=" + std::to_string(static_cast<unsigned>(error.code)));
    }
}

void ExpertHelper::cancel_round() noexcept { impl_->mailbox.cancel(); }

void ExpertHelper::cancel_and_drain() noexcept { impl_->mailbox.cancel_and_drain(); }
} // namespace ninfer::targets::qwen3_8_flash_next
