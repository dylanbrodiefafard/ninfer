#pragma once

#include "targets/qwen4/native_artifact.h"
#include "targets/qwen4/native_state.h"

namespace ninfer::targets::qwen4 {

// Borrowed startup-owned compact decoder workspace. Independent columns flatten W*B only
// across GR, matrix and MoE operations. Stateful Ops retain explicit row/slot/valid controls.
struct NativeDecoderViews {
    Tensor residual;     // BF16 [2560,4,W,B], input and output.
    Tensor ple_embedding;// BF16 [2560,W,B], represented host-table gathers.
    Tensor mixed,block;  // BF16 [2560,W,B].
    Tensor write_scale;  // BF16 [4,W,B].
    Tensor routes;       // I32 [10,W,B].
    Tensor route_weights;// FP32 [10,W,B].
    Tensor selected;     // I32 [2051,W,B].
    Tensor selected_count;// I32 [W,B].
    Tensor dflash_features; // Optional BF16 [12800,W,B].
    Tensor compact_rows; // I32 [B], exact identity 0..B-1 for compact feature capture.
};

// The production call supplies all48 distinct source layers. The actual first four-layer
// prefix is admitted only for bounded qualification on this GPU, sharing this very schedule;
// layers may not be repeated to synthesize a model. No embedding/head/sampling or publication
// happens here. The caller has materialized every QSA page and supplies disjoint committed /
// provisional recurrent state. One layer-major schedule serves the entire compact batch.
// full_prefill_slot>=0 is the host-proven single row with all W>16 columns live, outside
// decode/verification graphs. It selects the qualified chunked GDN prefill Op on that slot.
void enqueue_native_decoder(std::span<const NativeLayerWeights> layers,const NativePleWeights& ple,
    NativeState& state,const ops::QsaBatchControls& controls,int max_visible_keys,
    NativeDecoderViews views,bool record_prefix,WorkspaceArena& workspace,
    Tensor& qsa_workspace,cudaStream_t stream,int full_prefill_slot=-1);

} // namespace ninfer::targets::qwen4
