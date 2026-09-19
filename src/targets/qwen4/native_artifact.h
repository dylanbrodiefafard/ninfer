#pragma once

#include "artifact/materializer.h"
#include "targets/qwen4/mtp.h"
#include "targets/qwen4/dflash.h"
#include "targets/qwen4/vision.h"
#include "ninfer/ops/gated_delta_net_layer.h"
#include "ninfer/ops/ngram_embedding.h"
#include "ninfer/ops/ple.h"
#include "text/qwen/frontend_resources.h"

#include <map>
#include <optional>
#include <string>

namespace ninfer::targets::qwen4 {

struct NativeGrWeights { Tensor norm,inject; Weight down,up; };
struct NativeLayerWeights {
    NativeGrWeights attention_gr,moe_gr;
    ops::Qwen4ResidentSparseMoeWeights moe;
    ops::GatedDeltaNetLayerWeights gdn;
    ops::QsaVerifierWeights qsa;
};
struct NativePleWeights { Weight key,value; Tensor key_norm,query_norm,conv_norm,conv; };
enum class NativePleFormat { Nvfp4, Fp8 };
struct NativePleTable {
    NativePleFormat format=NativePleFormat::Nvfp4;
    ops::PleResidentNvfp4Table nvfp4;
    ops::PleResidentFp8Table fp8;
    std::uint16_t fp8_scale_bits=0;
};
// Borrow a complete already-resident encoded table; FP8 footer is separate from gather codes.
[[nodiscard]] NativePleTable native_ple_table_view(std::span<const std::byte>,NativePleFormat);

// Non-owning semantic views of the exact complete preview. Every referenced byte belongs to
// LoadedNativeModel; all programs/graphs must drain before that owner is destroyed.
struct NativeModelView {
    std::array<NativeLayerWeights,48> layers;
    NativePleWeights ple;
    NativePleTable ple_table;
    ops::NgramRowConfig ngram{248320,248044,0,1234,20000000};
    Weight token_embedding,output_head;
    NativeGrWeights final_gr;
    VisionWeights vision;
    MtpWeights mtp;
    std::optional<DFlashWeights> dflash;
};

struct NativeArtifactTensor {
    artifact::ObjectHandle handle;
    artifact::NumericFormat format;
    std::vector<std::uint64_t> shape;
};
// Complete metadata binding is independently usable before GPU allocation or host PLE locking.
// Names are startup-only binding keys, never an execution graph or runtime layer dispatch.
struct NativeArtifactPlan {
    artifact::MaterializationPlan materialization;
    std::map<std::string,NativeArtifactTensor> tensors;
    std::map<std::string,artifact::ObjectHandle> resources;
    NativePleFormat ple_format=NativePleFormat::Nvfp4;
    bool dflash=false;
    artifact::NumericFormat dflash_format=artifact::NumericFormat::BF16;
    text::qwen::FrontendResourcePlan frontend;
};
[[nodiscard]] NativeArtifactPlan bind_native_artifact(const artifact::Reader&);
void admit_native_device(const NativeArtifactPlan&,std::uint64_t available_bytes,
                         std::uint64_t runtime_reserve_bytes);

class LoadedNativeModel {
public:
    static std::unique_ptr<LoadedNativeModel> load(const std::filesystem::path&,DeviceContext&,
                                                  std::uint64_t runtime_reserve_bytes=0);
    const NativeModelView& view() const noexcept { return view_; }
    std::span<const std::byte> resource(std::string_view name) const;
    const artifact::MaterializationStats& stats() const noexcept { return backing_.stats(); }
    const text::qwen::FrontendResources& frontend_resources() const noexcept { return frontend_; }
private:
    artifact::MaterializedArtifact backing_;
    std::map<std::string,artifact::ObjectHandle> resources_;
    NativeModelView view_;
    text::qwen::FrontendResources frontend_;
};

} // namespace ninfer::targets::qwen4
