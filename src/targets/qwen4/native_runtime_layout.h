#pragma once

#include "targets/qwen4/native_runtime.h"
#include "core/layout.h"
#include "ninfer/ops/gated_residual.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::qwen4::native_layout {
constexpr int D=2560,F=10240,V=248320;
inline void validate(const NativeRuntimeConfig& c) {
    if(c.requests<1 || c.requests>4 || c.context_tokens<1 || c.context_tokens>262144 ||
       c.prefill_width<1 || c.prefill_width>4096 || c.verify_width<2 || c.verify_width>16 ||
       (c.mtp&&c.dflash) || (c.kv_dtype!=KvCacheStorage::BFloat16 && c.kv_dtype!=KvCacheStorage::Nvfp4) ||
       c.vision && (c.vision_patches<4 || c.vision_patches%4 || c.vision_segments<1 ||
         c.vision_segments>c.vision_patches/4 || c.vision_tokens_per_request<1))
        throw std::invalid_argument("Qwen4 native runtime startup profile");
    (void)NativeState::device_bytes({c.requests,c.context_tokens,c.kv_tokens,c.verify_width,
        c.kv_dtype==KvCacheStorage::Nvfp4?ops::QsaKvFormat::NVFP4G16:ops::QsaKvFormat::BF16});
}
inline int columns(const NativeRuntimeConfig& c) {return std::max(c.prefill_width,c.verify_width*c.requests);}
inline int visual_capacity(const NativeRuntimeConfig& c) {return std::min(c.context_tokens,c.vision_tokens_per_request);}
struct Shared {
    Tensor ids,positions,slots,valid,frontiers,counts,compact_rows;
    Tensor embedding,ple_embedding,residual,mixed,block,scale,routes,probabilities;
    Tensor logits,selected,selected_count,features,packed_rows;
    Tensor visual,visual_indices,patch_fp32,patch_bf16;
};
template<class Allocator> Shared allocate(Allocator& a,const NativeRuntimeConfig& c) {
    const int n=columns(c),b=c.requests;
    Shared s;
    s.ids=a.alloc(DType::I32,{n});s.positions=a.alloc(DType::I32,{3,n});
    s.slots=a.alloc(DType::I32,{b});s.valid=a.alloc(DType::I32,{b});
    s.frontiers=a.alloc(DType::I32,{b});s.counts=a.alloc(DType::I32,{b});
    s.compact_rows=a.alloc(DType::I32,{b});
    s.embedding=a.alloc(DType::BF16,{D,n});s.ple_embedding=a.alloc(DType::BF16,{D,n});
    s.residual=a.alloc(DType::BF16,{F,n});s.mixed=a.alloc(DType::BF16,{D,n});
    s.block=a.alloc(DType::BF16,{D,n});s.scale=a.alloc(DType::BF16,{4,n});
    s.routes=a.alloc(DType::I32,{10,n});s.probabilities=a.alloc(DType::FP32,{10,n});
    s.logits=a.alloc(DType::BF16,{V,n});s.selected=a.alloc(DType::I32,{2051,n});
    s.selected_count=a.alloc(DType::I32,{n});s.packed_rows=a.alloc(DType::U8,{2560,n});
    if(c.dflash) s.features=a.alloc(DType::BF16,{12800,n});
    if(c.vision) {
        // Current, retained, and prompt images must survive later media preparation.
        s.visual=a.alloc(DType::BF16,{D,visual_capacity(c),b,3});
        s.visual_indices=a.alloc(DType::I32,{n});
        s.patch_fp32=a.alloc(DType::FP32,{1536,c.vision_patches});
        s.patch_bf16=a.alloc(DType::BF16,{1536,c.vision_patches});
    }
    return s;
}
inline std::size_t shared_bytes(const NativeRuntimeConfig& c) {
    WorkspaceLayoutBuilder b;(void)allocate(b,c);return b.peak_bytes();
}
struct LayerPrecision {
    std::array<QType,4> gr;
    std::array<QType,4> mixer;
    ops::Qwen4ResidentSparseMoeStorageProfile moe;
};
struct Precision {
    std::array<LayerPrecision,48> layers;
    std::array<QType,2> ple,final;
    QType head;
};
inline Precision precision(const NativeModelView& model) {
    Precision p;
    for(int i=0;i<48;++i) {
        const auto& w=model.layers[i];auto& l=p.layers[i];
        l.gr={w.attention_gr.down.qtype,w.attention_gr.up.qtype,w.moe_gr.down.qtype,w.moe_gr.up.qtype};
        l.mixer=i%4==3?std::array{w.qsa.core_query_gate.qtype,w.qsa.core_key.qtype,w.qsa.core_value.qtype,w.qsa.output.qtype}:
            std::array{w.gdn.qkv.qtype,w.gdn.z.qtype,w.gdn.output.qtype,QType::BF16_CTRL};
        const auto& m=w.moe;
        l.moe={m.routed_gate.qtype,m.routed_up.qtype,m.routed_down.qtype,
            m.shared_gate_proj.qtype,m.shared_up.qtype,m.shared_down.qtype};
    }
    p.ple={model.ple.key.qtype,model.ple.value.qtype};
    p.final={model.final_gr.down.qtype,model.final_gr.up.qtype};p.head=model.output_head.qtype;return p;
}
inline QType qtype(artifact::NumericFormat format) {
    using N=artifact::NumericFormat;
    switch(format) {
        case N::BF16:return QType::BF16_CTRL;
        case N::NVFP4:return QType::NVFP4;
        case N::NVFP4_EXPERT_F32M:return QType::NVFP4_EXPERT_F32M;
        case N::FP8_E4M3FN_TENSOR_F32M:return QType::FP8_E4M3FN_TENSOR_F32M;
        default:throw std::invalid_argument("Qwen4 native matrix storage format");
    }
}
inline Precision precision(const NativeArtifactPlan& plan) {
    Precision p;
    const auto type=[&](const std::string& name){return qtype(plan.tensors.at(name).format);};
    const std::string main="model.language_model.";
    for(int i=0;i<48;++i) {
        const auto prefix=main+"layers."+std::to_string(i)+".";auto& l=p.layers[i];
        l.gr={type(prefix+"attn_hyper_connection.input_mix_weight_down.weight"),
            type(prefix+"attn_hyper_connection.input_mix_weight_up.weight"),
            type(prefix+"mlp_hyper_connection.input_mix_weight_down.weight"),
            type(prefix+"mlp_hyper_connection.input_mix_weight_up.weight")};
        if(i%4==3) l.mixer={type(prefix+"self_attn.q_proj.weight"),type(prefix+"self_attn.k_proj.weight"),
            type(prefix+"self_attn.v_proj.weight"),type(prefix+"self_attn.o_proj.weight")};
        else l.mixer={type(prefix+"linear_attn.in_proj_qkv.weight"),type(prefix+"linear_attn.in_proj_z.weight"),
            type(prefix+"linear_attn.out_proj.weight"),QType::BF16_CTRL};
        l.moe={type(prefix+"mlp.experts.gate_proj.weight"),type(prefix+"mlp.experts.up_proj.weight"),
            type(prefix+"mlp.experts.down_proj.weight"),type(prefix+"mlp.shared_expert.gate_proj.weight"),
            type(prefix+"mlp.shared_expert.up_proj.weight"),type(prefix+"mlp.shared_expert.down_proj.weight")};
    }
    p.ple={type(main+"layers.1.ple.key_proj.weight"),type(main+"layers.1.ple.value_proj.weight")};
    p.final={type(main+"hyper_connection_mixer.input_mix_weight_down.weight"),
             type(main+"hyper_connection_mixer.input_mix_weight_up.weight")};
    p.head=type("lm_head.weight");return p;
}
struct Scratch {std::size_t workspace=0,qsa=0;};
inline Scratch scratch(const Precision& p,const NativeRuntimeConfig& c) {
    Scratch result;
    const int n=columns(c);
    result.workspace=std::max({ops::ple_workspace_capacity_bytes(n,p.ple[0],p.ple[1]),
        ops::gated_residual_workspace_capacity_bytes(n,p.final[0],p.final[1]),
        ops::linear_workspace_capacity_bytes(p.head,V,D,ops::LinearPolicy::A16Only,1,n)});
    for(int i=0;i<48;++i) {
        const auto& l=p.layers[i];
        result.workspace=std::max({result.workspace,
            ops::gated_residual_workspace_capacity_bytes(n,l.gr[0],l.gr[1]),
            ops::gated_residual_workspace_capacity_bytes(n,l.gr[2],l.gr[3]),
            ops::qwen4_sparse_moe_resident_workspace_capacity_bytes(l.moe,n)});
        if(i%4!=3) result.workspace=std::max({result.workspace,
            ops::gated_delta_net_layer_batch_workspace_capacity_bytes(n,1,l.mixer[0],l.mixer[1],l.mixer[2]),
            ops::gated_delta_net_layer_workspace_capacity_bytes(c.prefill_width,l.mixer[0],l.mixer[1],l.mixer[2])});
        else for(int b=1;b<=c.requests;++b) {
            const int w=b==1?std::max(c.prefill_width,c.verify_width):c.verify_width;
            result.qsa=std::max(result.qsa,ops::qsa_verifier_workspace_bytes(w,b,
                l.mixer[0],l.mixer[1],l.mixer[2],l.mixer[3]));
        }
    }
    return result;
}
inline std::size_t graph_allowance(const NativeRuntimeConfig& c) {
    // Conservative driver/executable allowance, not arena bytes or a speed claim. Startup
    // checks actual graph consumption against it; immutable graph variants are exact B.
    return c.use_cuda_graph?std::size_t(c.requests)*(c.mtp||c.dflash?2:1)*12*1024*1024:0;
}
} // namespace ninfer::targets::qwen4::native_layout
