#pragma once

// Loads a qwen3.8-flash-next .ninfer artifact (docs/maintainer/qwen3.8-flash-next-artifact.md)
// into the storage behind a ModelView.

#include "model_view.h"
#include "ngram_table.h"
#include "runtime/expert_residency.h"

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/device.h"
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include <memory>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::targets::qwen3_8_flash_next {

/**
 * The resident model: every dense tensor in device memory (materialized from the artifact) except
 * a BF16 token embedding, which stays in device-mapped pinned host memory; every routed-expert
 * record in device-mapped pinned host memory (ExpertHostStore), which an ExpertResidency caches in
 * VRAM. Each layer's shared expert is a device-resident NVFP4 record. The six frontend resources
 * are retained as host bytes for the shared Qwen3.6 frontend.
 *
 * Construction reads from an opened artifact (the caller owns identity dispatch and the Reader,
 * which only needs to live through construction), accepts only qwen3.8-flash-next/f3-fp8-nvfp4,
 * binds every artifact object by its documented name, format, layout and shape (each fixed by the
 * profile), throws artifact::ArtifactError for another identity or a missing, extra or mismatched
 * object and propagates I/O, allocation and CUDA failures. A non-null `progress` receives the
 * materializer's "weights" phase and then an "experts" phase counting routed-record bytes copied
 * into the host store, both from the constructing thread. The views stay valid while the object
 * lives; it is not movable because ModelView points into its storage.
 */
class LoadedModel {
public:
    LoadedModel(const artifact::Reader& reader, DeviceContext& device,
                artifact::LoadProgress* progress = nullptr);

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;

    [[nodiscard]] const ModelView& view() const noexcept { return view_; }

    // The n-gram hash constants the checkpoint stores; the n-gram table must carry the same ones.
    [[nodiscard]] const NgramHashConstants& ngram_constants() const noexcept { return ngram_; }

    [[nodiscard]] const ExpertHostStore& experts() const noexcept { return *experts_; }

    [[nodiscard]] const qwen3_6::FrontendResources& frontend_resources() const noexcept {
        return frontend_;
    }

    // File, upload and staging counters of the dense materialization (the host store excluded).
    [[nodiscard]] const artifact::MaterializationStats& materialization_stats() const noexcept {
        return dense_.stats();
    }

private:
    artifact::MaterializedArtifact dense_;
    std::unique_ptr<ExpertHostStore> experts_;
    ModelView view_{};
    NgramHashConstants ngram_;
    qwen3_6::FrontendResources frontend_;
};

} // namespace ninfer::targets::qwen3_8_flash_next
