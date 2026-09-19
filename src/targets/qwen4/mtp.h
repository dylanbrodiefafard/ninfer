#pragma once

#include "artifact/materializer.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/qwen4_sparse_moe.h"

#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4 {

struct MtpGrWeights { Tensor norm,inject; Weight down,up; };
struct MtpWeights {
    Tensor embedding_norm,hidden_norm;
    Weight embedding,hidden;
    MtpGrWeights attention_gr,moe_gr,final_gr;
    ops::QsaVerifierWeights attention;
    ops::Qwen4ResidentSparseMoeWeights moe;
};

// Exact 1.6-GB native private-block owner; no target/embedding/head/PLE weights.
// The candidate's expert input scales are placeholders. This owner exposes no A4 policy.
class LoadedMtp {
public:
    static std::unique_ptr<LoadedMtp> load(const std::filesystem::path&,DeviceContext&);
    const MtpWeights& weights() const noexcept { return weights_; }
private:
    artifact::MaterializedArtifact backing_;
    std::vector<DeviceBuffer> controls_;
    MtpWeights weights_;
};

enum class MtpTraceStage { Stem,AttentionRead,AttentionWrite,Attention,AttentionInjected,
                           MoeRead,MoeWrite,MoeIds,MoeWeights,Moe,MoeInjected,FinalRead };
struct MtpTrace {
    void* context=nullptr;
    void (*capture)(void*,MtpTraceStage,const Tensor&,cudaStream_t)=nullptr;
};
struct MtpOutput { Tensor carried_hidden,logit_input; };

// One compact B<=4 native MTP compute schedule, with caller-owned paged BF16 state, controls,
// frontiers, private carry, frozen selections, and graphs. A16-only matrices. The audited draft
// profile freezes the COMPLETE step-zero selected IDs/count (vLLM+TokenSpeed), not SGLang's
// additional draft-token tail. Each step still appends private K/V/index/positions. No host
// readback, device allocation or synchronization occurs during execute. Returned views expire
// at next execution; no hidden request state lives in this scratch owner.
class MtpProgram {
public:
    MtpProgram(const MtpWeights&,int max_width,int max_batch);
    [[nodiscard]] static std::size_t device_bytes(int max_width,int max_batch);
    // Inputs BF16 [2560,W,B] and [10240,W,B]. Frozen pointers must both be present or absent;
    // their shape is I32 [2051,W,B] and [W,B]. Fresh selection is exposed below.
    [[nodiscard]] MtpOutput execute(const Tensor& embedding,const Tensor& hidden,
        const ops::QsaBatchControls&,ops::QsaPagedStateView,int max_visible_keys,
        const Tensor* frozen_ids,const Tensor* frozen_count,cudaStream_t,const MtpTrace* =nullptr);
    [[nodiscard]] Tensor selected_ids(int width,int batch) const;
    [[nodiscard]] Tensor selected_count(int width,int batch) const;
    [[nodiscard]] MtpOutput output(int width,int batch) const;
private:
    struct Storage;
    MtpProgram(const MtpWeights&,int,int,const Storage&);
    const MtpWeights& weights_;
    int max_width_,max_batch_;
    DeviceBuffer residual_,mixed_,block_,scale_,logit_,selected_,counts_,routes_,route_weights_,qsa_workspace_;
    WorkspaceArena workspace_;
};
} // namespace ninfer::targets::qwen4
