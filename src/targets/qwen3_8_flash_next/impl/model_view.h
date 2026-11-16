#pragma once

// Typed, non-owning views of every Qwen3.8-Flash-Next weight the forward pass reads. The loaded
// model (load/binder.h) owns the storage; a ModelView is valid while it lives. Matrix shapes below
// are [rows, columns] (Tensor ne {columns, rows}) unless stated otherwise.

#include "config.h"

#include "core/tensor.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/ple.h"

#include <array>
#include <optional>

namespace ninfer::targets::qwen3_8_flash_next {

// One Gated DeltaNet mixer.
struct GdnLayerWeights {
    // FP8 row-scaled q|k|v|z input projection [16384,2560], stored rows [2048,2048,6144,6144]
    // (gdn_input_proj).
    Weight query_key_value_z;
    // Depthwise causal convolution over q|k|v: BF16 Tensor ne {10240,4} (tap-major, channel
    // fastest), as causal_conv1d_silu_split reads it.
    Tensor convolution;
    // Packed a|b gate projection BF16 [96,2560]: rows 0..47 a, rows 48..95 b (bf16_projection).
    Tensor gate_projection;
    // FP32 [48] decay parameters of gdn_gating_packed.
    Tensor a_log;
    Tensor dt_bias;
    // Plain (not zero-centered) BF16 [128] gamma of the gated output RMSNorm.
    Tensor norm;
    // FP8 row-scaled output projection [2560,6144].
    Weight output;
};

// One QSA mixer.
struct QsaLayerWeights {
    // NVFP4 query/key/output-gate/value projection [13312,2560] (A16 only: no input divisor),
    // stored rows [6144,512,6144,512] with the checkpoint's per-head q|gate rows de-interleaved
    // (attn_input_proj).
    Weight query_key_gate_value;
    // Zero-centered BF16 [256] RMSNorm weights of the core query and key heads.
    Tensor query_norm;
    Tensor key_norm;
    // Indexer query [512,2560] and key [128,2560] projections, BF16.
    Tensor index_query;
    Tensor index_key;
    // Zero-centered BF16 [128] indexer norms (qsa_index_select).
    Tensor index_query_norm;
    Tensor index_key_norm;
    // FP8 row-scaled output projection [2560,6144].
    Weight output;
};

// The parts every layer has: two hyper-connections around the mixer and the MoE block, the packed
// MoE control projection, and the layer's experts.
struct LayerWeights {
    ops::HyperConnectionWeights mixer_connection;
    ops::HyperConnectionWeights moe_connection;
    // BF16 [544,2560]: 512 router rows, the shared-expert gate row, zero padding (moe_route).
    Tensor moe_control;
    // Everything moe_experts reads except expert_table, which is left empty here: the expert
    // cache's table column of the layer (ExpertResidency) is the routed records' address table.
    ops::MoeExpertBank experts;
};

// The n-gram embedding projections and injection of layer kPleLayer.
struct PleModelWeights {
    // BF16 key [10240,2560] and value [2560,2560] projections of the decoded n-gram embedding.
    Tensor key_projection;
    Tensor value_projection;
    ops::PleWeights inject;
};

// The five-layer Flash-Next DFlash2 drafter shares only the target's embedding and head.
struct DflashConvWeights {
    Tensor base_kernel;       // BF16 [2560,2,2], hidden fastest.
    Weight kernel_projection; // NVFP4 [640,2560].
};

struct DflashLayerWeights {
    Tensor input_norm;
    Weight query_key_value; // [6144,2560], rows q4096 | k1024 | v1024.
    Tensor query_norm;
    Tensor key_norm;
    Weight attention_output;
    DflashConvWeights attention_conv;
    Tensor post_attention_norm;
    Weight gate_up; // [15360,2560], gate then up.
    Weight down;
    DflashConvWeights mlp_conv;
};

struct DflashWeights {
    Weight feature_projection; // [2560,12800], means after layers 5,15,25,35,45.
    Tensor context_norm;
    std::array<DflashLayerWeights, 5> layers;
    Tensor final_norm;
    Weight hidden_projection;    // [256,2560].
    Tensor predecessor_codebook; // BF16 [256,248320], rank fastest.
    Tensor successor_codebook;
};

struct ModelView {
    // BF16 token embedding in device-mapped host memory and the FP8 row-scaled vocabulary head,
    // each [248320,2560].
    Weight embedding;
    Weight head;
    std::array<LayerWeights, kLayers> layers;
    std::array<GdnLayerWeights, kGdnLayers> gdn;
    std::array<QsaLayerWeights, kQsaLayers> qsa;
    PleModelWeights ple;
    // The model's final read-only hyper-connection read (write.data is null); there is no final
    // norm.
    ops::HyperConnectionWeights final_connection;
    std::optional<DflashWeights> dflash;
};

} // namespace ninfer::targets::qwen3_8_flash_next
