#pragma once

#include "ops/common/memory.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

enum class W8SmallTMmaScaleAccess : std::uint8_t {
    Direct,
    Shared,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct W8LinearGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 16) == 0);
    static_assert((InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows    = OutputRows;
    static constexpr std::int32_t kInputRows     = InputRows;
    static constexpr std::int32_t kGroupsPerRow  = InputRows / 32;
    static constexpr std::int32_t kCodeRowBytes  = InputRows;
    static constexpr std::int32_t kScaleRowBytes = kGroupsPerRow * sizeof(std::uint16_t);
};

template <int KWarps, int TileTokens, int MinBlocksPerSm, W8SmallTMmaScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg>
struct W8SmallTMmaSchedule {
    static_assert(KWarps == 4 || KWarps == 8);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = KWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / KWarps;
    static constexpr int kScaleBytesPerRow  = kGroupK / 16;
};

template <int TileTokens, int ActiveTokens>
using W8SmallTMmaDefaultSchedule = W8SmallTMmaSchedule<
    8, TileTokens, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
    (ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct)>;

using W8VocabularyProjectionGeometry = W8LinearGeometry<248320, 5120>;

inline constexpr std::int32_t kW8VocabularyFirstSmallT = 1;
inline constexpr std::int32_t kW8VocabularyLastSmallT  = 32;

template <class Geometry, int ActiveTokens>
struct W8LinearSmallTProductionSchedule;

template <int ActiveTokens>
struct W8LinearSmallTProductionSchedule<W8VocabularyProjectionGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kW8VocabularyFirstSmallT);
    static_assert(ActiveTokens <= kW8VocabularyLastSmallT);

    static constexpr int kTileTokens =
        ActiveTokens <= 8 ? 8 : (ActiveTokens <= 16 ? 16 : (ActiveTokens <= 24 ? 24 : 32));
    static constexpr auto kScaleAccess =
        ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    using Type = W8SmallTMmaSchedule<8, kTileTokens, 2, kScaleAccess>;
};

} // namespace ninfer::ops::detail
