#pragma once

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/vision_patch_merger.h"

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4 {

struct VisionGrid { std::int32_t temporal, height, width; };
struct VisionControl {
    std::int32_t patches = 0;
    std::vector<std::int32_t> positions; // [P,2], all H then all W.
    std::vector<std::int32_t> segments; // Each temporal grid plane is independent attention.
    std::vector<std::int32_t> position_indices; // [4,P], corner-fastest.
    std::vector<float> position_weights;
};

// Exact preview grid policy: 2x2 merge-major order, 48x48 align-corners interpolation.
// Grids name already prepared image/video patches; no resize or media decode happens here.
[[nodiscard]] VisionControl prepare_vision_control(std::span<const VisionGrid> grids);

struct VisionBlockWeights {
    Tensor norm1_weight, norm1_bias, norm2_weight, norm2_bias;
    Weight qkv, output, up, down;
    Tensor qkv_bias, output_bias, up_bias, down_bias;
};
struct VisionWeights {
    Weight patch;
    Tensor patch_bias, positions;
    std::array<VisionBlockWeights,27> blocks;
    ops::VisionPatchMergerWeights merger;
};

struct VisionTrace {
    // -1 is patch+position output; 0..26 are complete encoder-block outputs.
    // Diagnostic consumer enqueues its own copies on stream; it must not mutate the view.
    void* context = nullptr;
    void (*capture)(void*,std::int32_t,const Tensor&,cudaStream_t) = nullptr;
};

// Complete exact-source Vision artifact owner, deliberately not an Engine target.
class LoadedVision {
public:
    static std::unique_ptr<LoadedVision> load(const std::filesystem::path&, DeviceContext&);
    const VisionWeights& weights() const noexcept { return weights_; }
private:
    artifact::MaterializedArtifact backing_;
    VisionWeights weights_;
};

// Startup-fixed storage for the exact 27-block preview tower. The model and device outlive this
// instance; all consumers must drain before destruction or reconfiguration. configure() copies
// controls into instance-owned pinned staging and uploads outside capture. execute() launches
// only central Ops, without device allocation or synchronization;
// patches are caller-owned contiguous BF16 [1536,P] in source merge-major order. The returned
// BF16 [2560,P/4] is owned here and remains valid until the next execute/destruction. Each public
// Op output is the represented input to the next Op. Addresses remain stable for CUDA capture.
class VisionProgram {
public:
    VisionProgram(const VisionWeights&, std::int32_t max_patches, std::int32_t max_segments);
    [[nodiscard]] static std::size_t device_bytes(std::int32_t max_patches, std::int32_t max_segments);
    [[nodiscard]] static std::size_t pinned_bytes(std::int32_t max_patches, std::int32_t max_segments);
    void configure(const VisionControl&, cudaStream_t);
    [[nodiscard]] Tensor execute(const Tensor& patches, cudaStream_t, const VisionTrace* = nullptr);
    [[nodiscard]] Tensor encoder_output() const;
private:
    struct Storage;
    VisionProgram(const VisionWeights&,std::int32_t,std::int32_t,const Storage&);
    const VisionWeights& weights_;
    std::int32_t max_patches_, max_segments_, patches_=0, segments_=0;
    DeviceBuffer residual_, norm_, qkv_, attended_, projection_, up_, output_;
    DeviceBuffer positions_, segment_offsets_, position_indices_, position_weights_, merger_workspace_;
    PinnedHostBuffer control_staging_;
    WorkspaceArena workspace_;
};

} // namespace ninfer::targets::qwen4
