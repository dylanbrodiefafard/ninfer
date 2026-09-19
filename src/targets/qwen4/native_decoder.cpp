#include "targets/qwen4/native_decoder.h"

#include "targets/qwen4/dflash_features.h"
#include "core/nvtx.h"
#include "ninfer/ops/gated_residual.h"

#include <stdexcept>

namespace ninfer::targets::qwen4 {
void enqueue_native_decoder(std::span<const NativeLayerWeights> layers,const NativePleWeights& p,
    NativeState& state,const ops::QsaBatchControls& controls,int max_visible_keys,
    NativeDecoderViews views,bool record,WorkspaceArena& workspace,
    Tensor& qsa_workspace,cudaStream_t stream,int full_prefill_slot,NativePrefillPolicy prefill_policy) {
    const int width=views.residual.ne[2],batch=views.residual.ne[3],columns=width*batch;
    if((layers.size()!=48 && layers.size()!=4) || width<1 || batch<1 || batch>4 ||
       columns>4096 || (batch>1 && width>16) || (record && (width<2 || width>16)))
        throw std::invalid_argument("Qwen4 decoder compact geometry");
    if(full_prefill_slot>=0 && (batch!=1 || record || width<=16))
        throw std::invalid_argument("Qwen4 full-prefix prefill route");
    auto residual=views.residual.reshape({2560,4,columns});
    auto mixed=views.mixed.reshape({2560,columns});
    auto block=views.block.reshape({2560,columns});
    auto scale=views.write_scale.reshape({4,columns});
    auto routes=views.routes.reshape({10,columns});
    auto probability=views.route_weights.reshape({10,columns});
    auto records=record?state.records(width,batch):GdnReplayRecords{};
    auto ple_record=record?state.ple_records(width,batch):Tensor{};
    auto& before=state.gdn(false);auto& after=state.gdn(true);
    for(int layer=0;layer<int(layers.size());++layer) {
        const auto policy=native_prefill_policy(full_prefill_slot>=0?prefill_policy:NativePrefillPolicy::A16,layer);
        nvtx::ScopedRange layer_range(nvtx::Name::Qwen4Layer,nvtx::Category::Runtime,layer);
        if(layer==1) {
            // An external wait node must survive capture: the PLE rows change on every
            // replay, and their transfer is deliberately outside the compute graph.
            if(views.ple_ready) {
                cudaStreamCaptureStatus capture;
                CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
                CUDA_CHECK(cudaStreamWaitEvent(stream,views.ple_ready,
                    capture==cudaStreamCaptureStatusActive?cudaEventWaitExternal:0));
            }
            auto old_ple=state.ple(false),new_ple=state.ple(true);
            ops::ple_inject_batch(views.residual,views.ple_embedding,p.key,p.value,p.key_norm,
                p.query_norm,p.conv_norm,p.conv,old_ple,new_ple,controls.table_rows,
                controls.valid_columns,views.residual,workspace,ops::PleNormFormat::ZeroCenteredBf16,
                stream,record?&ple_record:nullptr);
        }
        const auto& w=layers[layer];const auto& a=w.attention_gr;
        ops::gated_residual_read_write(residual,a.norm,a.down,a.up,a.inject,mixed,scale,workspace,stream);
        if(views.dflash_features.data) {
            const DFlashFeatureSink sink{views.dflash_features,views.compact_rows,controls.valid_columns};
            sink.capture(layer,views.mixed,stream);
        }
        if(layer%4!=3) {
            const int recurrent=layer-layer/4;
            auto replay=record?records.layer(recurrent,batch):GdnReplayRecordLayer{};
            if(full_prefill_slot>=0) {
                // Eager full-prefix prefill retains the existing qualified chunked GDN route.
                // Compact masked decode/verification keeps device-selected state below.
                auto conv_in=before.conv_slot(recurrent,full_prefill_slot),conv_out=after.conv_slot(recurrent,full_prefill_slot);
                auto state_in=before.recurrent_slot(recurrent,full_prefill_slot),state_out=after.recurrent_slot(recurrent,full_prefill_slot);
                ops::gated_delta_net_layer(mixed,w.gdn,conv_in,conv_out,state_in,state_out,block,workspace,stream,policy.gdn);
            } else {
                ops::gated_delta_net_layer_batch(views.mixed,w.gdn,before.conv[recurrent],after.conv[recurrent],
                    before.recurrent[recurrent],after.recurrent[recurrent],controls.table_rows,
                    controls.valid_columns,views.block,workspace,stream,{},record?&replay:nullptr);
            }
        } else {
            auto cache=state.qsa(layer/4);
            ops::qsa_verifier(views.mixed,controls,max_visible_keys,w.qsa,cache,
                views.selected,views.selected_count,views.block,qsa_workspace,stream);
        }
        ops::gated_residual_inject(residual,block,scale,residual,stream);
        const auto& m=w.moe_gr;
        ops::gated_residual_read_write(residual,m.norm,m.down,m.up,m.inject,mixed,scale,workspace,stream);
        ops::qwen4_sparse_moe_resident(mixed,w.moe,routes,probability,block,workspace,stream,
                                      policy.routed,policy.shared);
        ops::gated_residual_inject(residual,block,scale,residual,stream);
    }
}
} // namespace ninfer::targets::qwen4
