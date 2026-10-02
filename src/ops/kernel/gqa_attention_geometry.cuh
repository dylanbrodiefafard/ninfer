#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int DecodeSplits     = 85 * DecodeSplitScale;
    // Small-T decode stages every (token, group-query) row of one KV head in at most three
    // 16-row MMA tiles: 27B (group of six) reaches T=8, 35B (group of eight) T=6.
    static constexpr int SmallTMaxTokens = 48 / GroupSize;
};

using Gqa27Geometry = GqaGeometry<24, 4, 1>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;

} // namespace ninfer::ops
