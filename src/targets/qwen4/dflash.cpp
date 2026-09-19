#include "targets/qwen4/dflash.h"

#include "artifact/typed_binding.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ninfer::targets::qwen4 {
namespace {
constexpr int kHidden = 2560;
constexpr int kVocabulary = 248320;
constexpr int kMask = 248077;
constexpr float kEpsilon = 1e-6F;
void observe(DFlashTrace trace, DFlashBoundary boundary, int layer, const Tensor& value,
             cudaStream_t stream) {
    if (trace.observe) trace.observe(trace.user, boundary, layer, value, stream);
}
Tensor plane(DeviceBuffer& buffer, int width, int columns) {
    return Tensor(buffer.p, DType::BF16, {width, columns});
}
std::size_t storage(int width, int append, int slots) {
    if (append < 1 || append > 4096 || slots < 1 || slots > 4)
        throw std::invalid_argument("Qwen4 DFlash requires append width 1..4096 and slots 1..4");
    return std::size_t(width) * std::max(append, 7) * slots * 2;
}
int pages(int capacity) {
    if (capacity < 1 || capacity > 262144)
        throw std::invalid_argument("Qwen4 DFlash context capacity must be 1..262144");
    return (capacity + 63) / 64;
}
std::size_t scratch(int capacity, int slots) {
    if (slots < 1 || slots > 4) throw std::invalid_argument("Qwen4 DFlash invalid slots");
    (void)pages(capacity);
    return std::max<std::size_t>({256, ops::bidirectional_gqa_attention_workspace_capacity_bytes(
        {0, std::uint32_t(capacity)}, 1, 7, slots, 256),
        ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,kVocabulary,kHidden,
            ops::LinearPolicy::A16Only,1,7*slots)});
}
void require_shape(const Tensor& t, DType dtype, int n0, int n1, int n2, const char* label) {
    if (t.dtype != dtype || !t.data || !t.is_contiguous() || t.ne[0] != n0 || t.ne[1] != n1 ||
        t.ne[2] != n2 || t.ne[3] != 1)
        throw std::invalid_argument(std::string("Qwen4 DFlash invalid ") + label);
}
void project(const Tensor& input, const Weight& weight, Tensor& output, int width,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_packed_sequences(input, weight, output, ops::LinearPolicy::A16Only,
                                 workspace, stream, width);
}
void require_shared_weight(const Weight& weight, const char* label) {
    if ((weight.qtype != QType::BF16_CTRL && weight.qtype != QType::FP8_E4M3FN_ROW_BF16S) ||
        weight.n != kVocabulary || weight.k != kHidden)
        throw std::invalid_argument(std::string("Qwen4 DFlash requires shared target BF16 or row-FP8 ") + label);
}
}

std::unique_ptr<LoadedDFlash> LoadedDFlash::load(const std::filesystem::path& path,
                                               DeviceContext& device) {
    artifact::Reader reader(path);
    const auto id = reader.identity();
    const bool nvfp4 = id == artifact::ArtifactIdentity{
        "qwen4/native-dflash-qualification", "pixelml-nvfp4-a16"};
    if (!nvfp4 && id != artifact::ArtifactIdentity{
            "qwen4/native-dflash-qualification", "pixelml-bf16-a16"})
        throw std::invalid_argument("Qwen4 DFlash requires the audited PixelML companion artifact");
    const auto format = nvfp4 ? artifact::NumericFormat::NVFP4 : artifact::NumericFormat::BF16;
    artifact::Binder binder(reader);
    std::map<std::string, artifact::ObjectHandle> handles;
    std::map<std::string, float> divisors;
    auto bind_norm = [&](const std::string& name, int width) {
        handles.emplace(name, artifact::bind_device_tensor(binder, name,
            artifact::NumericFormat::BF16, {std::uint64_t(width)}));
    };
    auto bind_matrix = [&](const std::string& name, int n, int k) {
        auto handle = artifact::bind_device_tensor(binder, name, format,
                                                   {std::uint64_t(n), std::uint64_t(k)});
        handles.emplace(name, handle);
        if (nvfp4) {
            const std::array<std::uint64_t,2> shape{std::uint64_t(n), std::uint64_t(k)};
            const auto layout = artifact::block_scale_geometry(format, shape);
            float divisor;
            std::memcpy(&divisor, binder.payload(handle).data.data() + layout.weight_divisor_offset, 4);
            if (!(divisor > 0) || !std::isfinite(divisor))
                throw std::invalid_argument("Qwen4 DFlash invalid NVFP4 weight divisor");
            divisors.emplace(name, divisor);
        }
    };
    bind_matrix("fc.weight", 2560, 12800);
    bind_norm("hidden_norm.weight", 2560);
    bind_norm("norm.weight", 2560);
    constexpr std::array<std::tuple<const char*,int,int>,7> matrices{{
        {"self_attn.q_proj.weight",6144,2560}, {"self_attn.k_proj.weight",512,2560},
        {"self_attn.v_proj.weight",512,2560}, {"self_attn.o_proj.weight",2560,6144},
        {"mlp.gate_proj.weight",7680,2560}, {"mlp.up_proj.weight",7680,2560},
        {"mlp.down_proj.weight",2560,7680}}};
    for (int layer=0; layer<5; ++layer) {
        const auto p = "layers." + std::to_string(layer) + ".";
        bind_norm(p+"input_layernorm.weight",2560);
        bind_norm(p+"post_attention_layernorm.weight",2560);
        bind_norm(p+"self_attn.q_norm.weight",256);
        bind_norm(p+"self_attn.k_norm.weight",256);
        for (auto [name,n,k] : matrices) bind_matrix(p+name,n,k);
    }
    auto result = std::unique_ptr<LoadedDFlash>(new LoadedDFlash);
    result->backing_ = artifact::materialize(reader,binder.finish(),device);
    auto norm = [&](const std::string& name,int width) {
        return artifact::materialized_tensor(result->backing_,handles.at(name),
                                              artifact::NumericFormat::BF16,{width});
    };
    auto matrix = [&](const std::string& name,int n,int k) {
        const auto handle = handles.at(name);
        if (!nvfp4) return artifact::materialized_weight(result->backing_,handle,format,n,k);
        const std::array<std::uint64_t,2> shape{std::uint64_t(n),std::uint64_t(k)};
        const auto layout = artifact::block_scale_geometry(format,shape);
        const auto* data = static_cast<const std::byte*>(result->backing_.device_data(handle));
        Weight w{};
        w.payload=w.qdata=data; w.payload_bytes=layout.encoded_bytes;
        w.scales=data+layout.scale_plane_offset; w.qtype=QType::NVFP4;
        w.layout=QuantLayout::BlockScaleK16M128x4; w.scale_dtype=DType::FP8_E4M3FN;
        w.group_size=w.group=16; w.ndim=2;
        w.n=w.shape[0]=w.padded_shape[0]=n;
        w.k=w.shape[1]=w.padded_shape[1]=k;
        w.weight_scale_divisor=divisors.at(name);
        // No A4 route is called: this is an explicit A16 fixture, not input calibration.
        w.input_scale_divisor=1.0F;
        return w;
    };
    auto& w=result->weights_;
    w.feature=matrix("fc.weight",2560,12800);
    w.hidden_norm=norm("hidden_norm.weight",2560); w.final_norm=norm("norm.weight",2560);
    for(int layer=0;layer<5;++layer) {
        const auto p="layers."+std::to_string(layer)+"."; auto& b=w.layers[layer];
        b.input_norm=norm(p+"input_layernorm.weight",2560);
        b.post_norm=norm(p+"post_attention_layernorm.weight",2560);
        b.query_norm=norm(p+"self_attn.q_norm.weight",256);
        b.key_norm=norm(p+"self_attn.k_norm.weight",256);
        std::array<Weight*,7> destinations{&b.query,&b.key,&b.value,&b.output,&b.gate,&b.up,&b.down};
        for(std::size_t i=0;i<matrices.size();++i) {
            auto [name,n,k]=matrices[i]; *destinations[i]=matrix(p+name,n,k);
        }
    }
    return result;
}

struct DFlashProgram::Storage {
    std::size_t hidden,query,key,ffn,ids,embedding,workspace;
    Storage(int capacity,int append,int slots):hidden(storage(2560,append,slots)),
        query(storage(6144,append,slots)),key(storage(512,append,slots)),ffn(storage(7680,append,slots)),
        ids(std::size_t(7)*slots*4),embedding(storage(2560,7,slots)),workspace(scratch(capacity,slots)) {}
    std::size_t bytes() const {return hidden*4+query*3+key*3+ffn*3+ids*2+embedding+workspace;}
};
std::size_t DFlashProgram::device_bytes(int capacity,int append,int slots) {return Storage(capacity,append,slots).bytes();}
DFlashProgram::DFlashProgram(const DFlashWeights& weights,int capacity,int append,int slots,
    const std::array<PagedKVBatchLayerView,5>& caches)
    :DFlashProgram(weights,capacity,append,slots,caches,Storage(capacity,append,slots)) {}
DFlashProgram::DFlashProgram(const DFlashWeights& weights,int capacity,int append,int slots,
    const std::array<PagedKVBatchLayerView,5>& caches,const Storage& s)
    : weights_(weights),context_capacity_(capacity),append_width_(append),slots_(slots),
      caches_(caches),residual_(s.hidden),norm_(s.hidden),projected_(s.hidden),context_(s.hidden),
      q_raw_(s.query),q_(s.query),k_raw_(s.key),k_(s.key),v_(s.key),attention_(s.query),
      gate_(s.ffn),up_(s.ffn),activated_(s.ffn),ids_(s.ids),positions_(s.ids),
      embeddings_(s.embedding),workspace_(s.workspace) {
    for(const auto& cache:caches_) {
        if(cache.dtype!=DType::BF16 || cache.head_dim!=256 || cache.num_kv_heads!=2 ||
           cache.quant_group!=0 || cache.k_pages.ne[0]!=256 || cache.k_pages.ne[1]!=64 ||
           cache.k_pages.ne[3]!=2 || cache.k_pages.ne[2]<1 ||
           !std::equal(std::begin(cache.v_pages.ne),std::end(cache.v_pages.ne),std::begin(cache.k_pages.ne)) ||
           cache.k_pages.dtype!=DType::BF16 || cache.v_pages.dtype!=DType::BF16 ||
           !cache.k_pages.data || !cache.v_pages.data || !cache.block_tables.data ||
           !cache.k_pages.is_contiguous() || !cache.v_pages.is_contiguous() ||
           cache.block_tables.ne[0]<pages(capacity) || cache.block_tables.ne[1]!=slots ||
           cache.block_tables.dtype!=DType::I32 || !cache.block_tables.is_contiguous())
            throw std::invalid_argument("Qwen4 DFlash requires caller-owned BF16 head-major paged context");
    }
}

PagedKVBatchLayerView DFlashProgram::cache(int layer) const {
    if(layer<0 || layer>=5) throw std::invalid_argument("Qwen4 DFlash invalid cache layer");
    return caches_[layer];
}

void DFlashProgram::append_accepted_context(const Tensor& features,const Tensor& rope_positions,
    const Tensor& cache_positions,const Tensor& counts,const Tensor& slots,
    ops::KVCacheAppendPrefixExecutionEnvelope envelope,cudaStream_t stream,DFlashTrace trace) {
    const int width=features.ne[1],batch=features.ne[2];
    if(width<1 || width>append_width_ || batch<1 || batch>slots_)
        throw std::invalid_argument("Qwen4 DFlash context append exceeds startup capacity");
    const int columns=width*batch;
    require_shape(features,DType::BF16,12800,width,batch,"features");
    require_shape(rope_positions,DType::I32,width,batch,1,"context RoPE positions");
    require_shape(cache_positions,DType::I32,width,batch,1,"context cache positions");
    require_shape(counts,DType::I32,batch,1,1,"accepted counts");
    require_shape(slots,DType::I32,batch,1,1,"slots");
    auto projected=plane(projected_,2560,columns),context=plane(context_,2560,columns);
    project(features.view({12800,columns}),weights_.feature,projected,width,workspace_,stream);
    observe(trace,DFlashBoundary::FeatureProjection,-1,projected,stream);
    ops::rmsnorm(projected,weights_.hidden_norm,kEpsilon,false,context,stream);
    observe(trace,DFlashBoundary::FusedContext,-1,context,stream);
    for(int layer=0;layer<5;++layer) {
        const auto& w=weights_.layers[layer];
        auto raw=plane(k_raw_,512,columns),key=plane(k_,512,columns),value=plane(v_,512,columns);
        project(context,w.key,raw,width,workspace_,stream);
        observe(trace,DFlashBoundary::ContextKeyProjection,layer,raw,stream);
        project(context,w.value,value,width,workspace_,stream);
        observe(trace,DFlashBoundary::ContextValue,layer,value,stream);
        auto raw_heads=raw.view({256,2,columns}),key_heads=key.view({256,2,columns});
        ops::rmsnorm(raw_heads,w.key_norm,kEpsilon,false,key_heads,stream);
        observe(trace,DFlashBoundary::ContextKeyNorm,layer,key_heads,stream);
        ops::rope(rope_positions.view({columns}),256,1e7F,key_heads,stream);
        observe(trace,DFlashBoundary::ContextKey,layer,key_heads,stream);
        ops::kv_cache_append_prefix(key.view({256,2,width,batch}),value.view({256,2,width,batch}),
            cache_positions,counts,slots,envelope,cache(layer),stream);
    }
}

Tensor DFlashProgram::draft_embeddings(const Tensor& embeddings,const Tensor& rope_positions,
    const Tensor& lengths,const Tensor& valid,const Tensor& slots,cudaStream_t stream,DFlashTrace trace) {
    const int width=embeddings.ne[1],batch=embeddings.ne[2];
    if(width<1 || width>7 || batch<1 || batch>slots_)
        throw std::invalid_argument("Qwen4 DFlash draft block exceeds startup capacity");
    const int columns=width*batch;
    require_shape(embeddings,DType::BF16,2560,width,batch,"query embeddings");
    require_shape(rope_positions,DType::I32,width,batch,1,"query RoPE positions");
    require_shape(lengths,DType::I32,batch,1,1,"context lengths");
    require_shape(valid,DType::I32,batch,1,1,"valid query counts");
    require_shape(slots,DType::I32,batch,1,1,"slots");
    auto residual=plane(residual_,2560,columns),normalized=plane(norm_,2560,columns);
    CUDA_CHECK(cudaMemcpyAsync(residual.data,embeddings.data,residual.bytes(),cudaMemcpyDeviceToDevice,stream));
    for(int layer=0;layer<5;++layer) {
        const auto& w=weights_.layers[layer];
        ops::rmsnorm(residual,w.input_norm,kEpsilon,false,normalized,stream);
        observe(trace,DFlashBoundary::InputNorm,layer,normalized,stream);
        auto qr=plane(q_raw_,6144,columns),kr=plane(k_raw_,512,columns),value=plane(v_,512,columns);
        project(normalized,w.query,qr,width,workspace_,stream);
        observe(trace,DFlashBoundary::QueryProjection,layer,qr,stream);
        project(normalized,w.key,kr,width,workspace_,stream);
        observe(trace,DFlashBoundary::KeyProjection,layer,kr,stream);
        project(normalized,w.value,value,width,workspace_,stream);
        observe(trace,DFlashBoundary::Value,layer,value,stream);
        auto query=Tensor(q_.p,DType::BF16,{256,24,columns});
        auto key=Tensor(k_.p,DType::BF16,{256,2,columns});
        ops::rmsnorm(qr.view({256,24,columns}),w.query_norm,kEpsilon,false,query,stream);
        observe(trace,DFlashBoundary::QueryNorm,layer,query,stream);
        ops::rmsnorm(kr.view({256,2,columns}),w.key_norm,kEpsilon,false,key,stream);
        observe(trace,DFlashBoundary::KeyNorm,layer,key,stream);
        ops::rope(rope_positions.view({columns}),256,1e7F,query,key,stream);
        observe(trace,DFlashBoundary::Query,layer,query,stream);
        observe(trace,DFlashBoundary::Key,layer,key,stream);
        auto attention=Tensor(attention_.p,DType::BF16,{256,24,width,batch});
        ops::bidirectional_gqa_attention(query.view({256,24,width,batch}),
            key.view({256,2,width,batch}),value.view({256,2,width,batch}),lengths,valid,slots,
            0.0625F,cache(layer),{0,std::uint32_t(context_capacity_)},workspace_,attention,stream);
        observe(trace,DFlashBoundary::Attention,layer,attention,stream);
        auto projected=plane(projected_,2560,columns);
        project(attention.view({6144,columns}),w.output,projected,width,workspace_,stream);
        observe(trace,DFlashBoundary::AttentionProjection,layer,projected,stream);
        ops::residual_add(projected,residual,stream);
        observe(trace,DFlashBoundary::AttentionResidual,layer,residual,stream);
        ops::rmsnorm(residual,w.post_norm,kEpsilon,false,normalized,stream);
        observe(trace,DFlashBoundary::PostNorm,layer,normalized,stream);
        auto gate=plane(gate_,7680,columns),up=plane(up_,7680,columns),active=plane(activated_,7680,columns);
        project(normalized,w.gate,gate,width,workspace_,stream);
        observe(trace,DFlashBoundary::Gate,layer,gate,stream);
        project(normalized,w.up,up,width,workspace_,stream);
        observe(trace,DFlashBoundary::Up,layer,up,stream);
        ops::silu_mul(gate,up,active,stream);
        observe(trace,DFlashBoundary::Activated,layer,active,stream);
        project(active,w.down,projected,width,workspace_,stream);
        observe(trace,DFlashBoundary::Down,layer,projected,stream);
        ops::residual_add(projected,residual,stream);
        observe(trace,DFlashBoundary::LayerOutput,layer,residual,stream);
    }
    ops::rmsnorm(residual,weights_.final_norm,kEpsilon,false,normalized,stream);
    observe(trace,DFlashBoundary::FinalOutput,-1,normalized,stream);
    return normalized.view({2560,width,batch});
}

void DFlashProgram::draft(const Tensor& anchors,const Tensor& anchor_positions,
    const Tensor& lengths,const Tensor& valid,
    const Tensor& slots,const Weight& embedding,const Weight& head,Tensor& logits,
    cudaStream_t stream,DFlashTrace trace) {
    const int width=logits.ne[1],batch=logits.ne[2];
    if(width<1 || width>7 || batch<1 || batch>slots_)
        throw std::invalid_argument("Qwen4 DFlash draft shape exceeds startup capacity");
    const int columns=width*batch;
    require_shape(logits,DType::BF16,kVocabulary,width,batch,"logits");
    require_shape(anchors,DType::I32,batch,1,1,"anchors");
    require_shape(anchor_positions,DType::I32,batch,1,1,"anchor positions");
    require_shape(lengths,DType::I32,batch,1,1,"context lengths");
    require_shape(valid,DType::I32,batch,1,1,"valid query counts");
    require_shape(slots,DType::I32,batch,1,1,"slots");
    require_shared_weight(embedding,"embedding"); require_shared_weight(head,"head");
    auto ids=Tensor(ids_.p,DType::I32,{width,batch});
    auto positions=Tensor(positions_.p,DType::I32,{width,batch});
    ops::prepare_masked_block(anchors,anchor_positions,valid,kMask,ids,positions,stream);
    auto emb=plane(embeddings_,2560,columns);
    ops::embedding(ids.view({columns}),embedding,emb,stream);
    const auto hidden=draft_embeddings(emb.view({2560,width,batch}),positions,lengths,valid,slots,stream,trace);
    auto flat_logits=logits.view({kVocabulary,columns});
    project(hidden.view({2560,columns}),head,flat_logits,width,workspace_,stream);
}

} // namespace ninfer::targets::qwen4
