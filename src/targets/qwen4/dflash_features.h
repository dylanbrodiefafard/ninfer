#pragma once

#include "ninfer/ops/scatter.h"

#include <array>
#include <stdexcept>

namespace ninfer::targets::qwen4 {

// Publisher tap labels [3,15,23,35,43] refer to the learned attention-GR contraction at the
// NEXT layer boundary, not that prior layer's raw four-stream residual or its FFN block input.
// The target schedule invokes capture immediately after attention-GR read, before the mixer.
// Packed BF16 [12800,W,C] storage and I32 [B] lane/valid controls are caller-owned and stable
// through queued work. Invalid columns and other slots are unchanged. No allocation or sync.
struct DFlashFeatureSink {
    Tensor packed;
    Tensor slots;
    Tensor valid_columns;

    void capture(int layer, const Tensor& block_input, cudaStream_t stream) const {
        constexpr std::array<int,5> boundaries{4,16,24,36,44};
        for(int tap=0;tap<5;++tap) if(layer==boundaries[tap]) {
            if(packed.dtype!=DType::BF16 || packed.ne[0]!=12800 || packed.ne[3]!=1 ||
               block_input.dtype!=DType::BF16 || block_input.ne[0]!=2560 ||
               block_input.ne[1]!=packed.ne[1] || block_input.ne[3]!=1)
                throw std::invalid_argument("Qwen4 DFlash feature capture shape mismatch");
            Tensor destination=packed.slice(0,tap*2560,2560);
            ops::scatter_bf16_batch(block_input,slots,valid_columns,destination,stream);
            return;
        }
    }
};

} // namespace ninfer::targets::qwen4
