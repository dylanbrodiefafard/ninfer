#pragma once

#include "ninfer/ops/gated_delta_net_layer.h"
#include "ninfer/ops/qwen4_sparse_moe.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen4 {

// Exact preview candidate recipes, not a model graph or a model-quality admission.
// Encoded as one byte in native-prefill-policy. Decode, verification, QSA, GR, PLE,
// endpoints and private draft blocks remain A16. Only full, unmasked C1 prefill opts in.
enum class NativePrefillPolicy : std::uint8_t {
    A16 = 0, SelectiveA8 = 1, RoutedA4 = 2, RoutedA4SelectiveA8 = 3
};
inline bool selective_a8(NativePrefillPolicy p) {
    return p==NativePrefillPolicy::SelectiveA8 || p==NativePrefillPolicy::RoutedA4SelectiveA8;
}
struct NativeLayerPolicy {
    ops::GatedDeltaNetProjectionPolicy gdn;
    ops::LinearPolicy routed=ops::LinearPolicy::A16Only;
    ops::Qwen4SharedExpertPolicy shared;
};
inline NativeLayerPolicy native_prefill_policy(NativePrefillPolicy p,int layer) {
    if(static_cast<unsigned>(p)>3 || layer<0 || layer>=48)
        throw std::invalid_argument("invalid exact native prefill policy");
    NativeLayerPolicy result;
    if(p==NativePrefillPolicy::RoutedA4 || p==NativePrefillPolicy::RoutedA4SelectiveA8)
        result.routed=ops::LinearPolicy::AllowA4;
    if(selective_a8(p)) {
        if(layer==0) result.gdn.z=ops::LinearPolicy::AllowA8;
        if(layer==0 || layer==3)
            result.shared={ops::LinearPolicy::AllowA8,ops::LinearPolicy::AllowA8,ops::LinearPolicy::AllowA8};
    }
    return result;
}
} // namespace ninfer::targets::qwen4
