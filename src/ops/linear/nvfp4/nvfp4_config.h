#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

enum class Nvfp4ScaleAccess : std::uint8_t {
    StagedRaw,
    Direct,
};

enum class Nvfp4CodeCache : std::uint8_t {
    Default,   // plain load (compiler / L1 eligible)
    Streaming, // ld.global.cs — do not retain in L2 (weight one-shot)
    L2Cached,  // ld.global.cg — bypass L1, keep in L2
};

enum class Nvfp4SmallTActivationAccess : std::uint8_t {
    PairStream,
    TokenPacked,
    SharedPhase,
};

enum class Nvfp4SmallTBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4GemvGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int MinBlocksPerSm>
struct Nvfp4GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane, int TokenTile,
          int AccumulatorChains, Nvfp4SmallTActivationAccess ActivationAccess,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int PhaseUnroll,
          Nvfp4SmallTBlockOrder BlockOrder, int MinBlocksPerSm>
struct Nvfp4SmallTSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4 || PhaseUnroll == 10);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

using Nvfp4AttnInputGeometry     = Nvfp4GemvGeometry<14336, 5120>;
using Nvfp4GdnInputGeometry      = Nvfp4GemvGeometry<16384, 5120>;
using Nvfp4MlpGateUpGeometry     = Nvfp4GemvGeometry<34816, 5120>;
using Nvfp4Residual6144Geometry  = Nvfp4GemvGeometry<5120, 6144>;
using Nvfp4Residual17408Geometry = Nvfp4GemvGeometry<5120, 17408>;
using Nvfp4DflashFeatureGeometry = Nvfp4GemvGeometry<5120, 25600>;
using Nvfp4DflashQkvGeometry     = Nvfp4GemvGeometry<6144, 5120>;
using Nvfp4DflashAttnOutGeometry = Nvfp4GemvGeometry<5120, 4096>;
using Nvfp4DflashConvProjGeometry = Nvfp4GemvGeometry<1280, 5120>;
using Nvfp4DflashSelectorGeometry = Nvfp4GemvGeometry<256, 5120>;
using Nvfp4MtpFcGeometry         = Nvfp4GemvGeometry<5120, 10240>;

using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation10240Geometry = Nvfp4ActivationGeometry<10240>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;

enum class Nvfp4Problem : std::uint8_t {
    AttnInput,
    GdnInput,
    MlpGateUp,
    Residual6144,
    Residual17408,
    DflashFeature,
    DflashQkv,
    DflashAttnOut,
    DflashConvProj,
    DflashSelector,
    MtpFc,
};

inline constexpr bool is_nvfp4_a16_only_problem(Nvfp4Problem problem) {
    return problem == Nvfp4Problem::DflashFeature || problem == Nvfp4Problem::DflashQkv ||
           problem == Nvfp4Problem::DflashAttnOut || problem == Nvfp4Problem::DflashConvProj ||
           problem == Nvfp4Problem::DflashSelector;
}

inline constexpr bool is_nvfp4_linear_problem(std::int32_t output_rows, std::int32_t input_rows) {
    return (output_rows == Nvfp4AttnInputGeometry::kOutputRows &&
            input_rows == Nvfp4AttnInputGeometry::kInputRows) ||
           (output_rows == Nvfp4GdnInputGeometry::kOutputRows &&
            input_rows == Nvfp4GdnInputGeometry::kInputRows) ||
           (output_rows == Nvfp4MlpGateUpGeometry::kOutputRows &&
            input_rows == Nvfp4MlpGateUpGeometry::kInputRows) ||
           (output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
            input_rows == Nvfp4Residual6144Geometry::kInputRows) ||
           (output_rows == Nvfp4Residual17408Geometry::kOutputRows &&
            input_rows == Nvfp4Residual17408Geometry::kInputRows) ||
           (output_rows == Nvfp4DflashFeatureGeometry::kOutputRows &&
            input_rows == Nvfp4DflashFeatureGeometry::kInputRows) ||
           (output_rows == Nvfp4DflashQkvGeometry::kOutputRows &&
            input_rows == Nvfp4DflashQkvGeometry::kInputRows) ||
           (output_rows == Nvfp4DflashAttnOutGeometry::kOutputRows &&
            input_rows == Nvfp4DflashAttnOutGeometry::kInputRows) ||
           (output_rows == Nvfp4DflashConvProjGeometry::kOutputRows &&
            input_rows == Nvfp4DflashConvProjGeometry::kInputRows) ||
           (output_rows == Nvfp4DflashSelectorGeometry::kOutputRows &&
            input_rows == Nvfp4DflashSelectorGeometry::kInputRows) ||
           (output_rows == Nvfp4MtpFcGeometry::kOutputRows &&
            input_rows == Nvfp4MtpFcGeometry::kInputRows);
}

inline Nvfp4Problem resolve_nvfp4_problem(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Nvfp4AttnInputGeometry::kOutputRows &&
        input_rows == Nvfp4AttnInputGeometry::kInputRows) {
        return Nvfp4Problem::AttnInput;
    }
    if (output_rows == Nvfp4GdnInputGeometry::kOutputRows &&
        input_rows == Nvfp4GdnInputGeometry::kInputRows) {
        return Nvfp4Problem::GdnInput;
    }
    if (output_rows == Nvfp4MlpGateUpGeometry::kOutputRows &&
        input_rows == Nvfp4MlpGateUpGeometry::kInputRows) {
        return Nvfp4Problem::MlpGateUp;
    }
    if (output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
        input_rows == Nvfp4Residual6144Geometry::kInputRows) {
        return Nvfp4Problem::Residual6144;
    }
    if (output_rows == Nvfp4Residual17408Geometry::kOutputRows &&
        input_rows == Nvfp4Residual17408Geometry::kInputRows) {
        return Nvfp4Problem::Residual17408;
    }
    if (output_rows == Nvfp4DflashFeatureGeometry::kOutputRows &&
        input_rows == Nvfp4DflashFeatureGeometry::kInputRows) {
        return Nvfp4Problem::DflashFeature;
    }
    if (output_rows == Nvfp4DflashQkvGeometry::kOutputRows &&
        input_rows == Nvfp4DflashQkvGeometry::kInputRows) {
        return Nvfp4Problem::DflashQkv;
    }
    if (output_rows == Nvfp4DflashAttnOutGeometry::kOutputRows &&
        input_rows == Nvfp4DflashAttnOutGeometry::kInputRows) {
        return Nvfp4Problem::DflashAttnOut;
    }
    if (output_rows == Nvfp4DflashConvProjGeometry::kOutputRows &&
        input_rows == Nvfp4DflashConvProjGeometry::kInputRows) {
        return Nvfp4Problem::DflashConvProj;
    }
    if (output_rows == Nvfp4DflashSelectorGeometry::kOutputRows &&
        input_rows == Nvfp4DflashSelectorGeometry::kInputRows) {
        return Nvfp4Problem::DflashSelector;
    }
    if (output_rows == Nvfp4MtpFcGeometry::kOutputRows &&
        input_rows == Nvfp4MtpFcGeometry::kInputRows) {
        return Nvfp4Problem::MtpFc;
    }
    throw std::invalid_argument("unsupported NVFP4 problem");
}

// RTX 5090 cold-cache winner among the measured decode schedules.
template <class Geometry>
struct Nvfp4LinearDecodeProductionSchedule {
    using Type =
        Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
};

// MTP fc [5120,10240] T=1 AR: N is residual-class. Four-warp CTAs match the measured SmallT
// occupancy for this N (the generic 8-warp decode schedule is the wide-N AttnInput winner).
template <>
struct Nvfp4LinearDecodeProductionSchedule<Nvfp4MtpFcGeometry> {
    using Type =
        Nvfp4GemvSchedule<4, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
};

inline constexpr std::int32_t kNvfp4FirstSmallT = 2;
inline constexpr std::int32_t kNvfp4LastSmallT  = 32;
// Production AllowA4 W4A4 cutovers (RTX 5090). T=1 stays GEMV: the W4A4 MMA tile is
// BlockM=32 and T=1 residual is not a legal decode path (2× vs the A4 oracle).
inline constexpr std::int32_t kNvfp4FirstW4a4AttnInput     = 4;
inline constexpr std::int32_t kNvfp4FirstW4a4GdnInput      = 3;
inline constexpr std::int32_t kNvfp4FirstW4a4MlpGateUp     = 2;
inline constexpr std::int32_t kNvfp4FirstW4a4Residual6144  = 5;
inline constexpr std::int32_t kNvfp4FirstW4a4Residual17408 = 3;
inline constexpr std::int32_t kNvfp4FirstW4a4MtpFc         = 8;
// Tree/chain verify is W<=16. Fused GDN conv stays A16 SmallT on that range (parent-indexed
// FP32 epilogue), including C>1 (one launch, grid.x=B). W4A4 Materialized compose is prefill.
inline constexpr std::int32_t kNvfp4LastPackedGdnConvSmallT = 16;

// RTX 5090 cold-cache winners for contiguous Linear output. T=2..4 amortizes activation loads
// through shared staging; T=5..32 keeps one packed activation tile per warp. The warp-count changes
// are measured occupancy/register crossovers, not semantic frontiers.
template <class Geometry, int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 13 ? 16 : 8);
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// G1's wider N benefits from keeping four warps per CTA throughout the A16 policy boundary. Only
// T=2 amortizes activation traffic enough for shared staging to win.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    // Same 8-warp / StagedRaw reduction as T=1 GDN GEMV. 4 accumulator chains match
    // decode through packed verify (T<=16); the A16-only tail keeps one chain.
    static constexpr int kWarpsPerCta       = 8;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr int kAccumulatorChains = ActiveTokens <= 16 ? 4 : 1;
    static constexpr auto kActivationAccess = ActiveTokens == 2
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, kAccumulatorChains,
                            kActivationAccess, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default,
                            1, Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// At N=5120, R1 needs the larger CTA only for the last three A16 token counts. The unoptimized
// A16-only tail keeps the established generic schedule.
// T=2..4: PhaseUnroll=4 is bit-exact; cold +10% / warm ~flat (geometry_sweep_v5).
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 16 ? (ActiveTokens >= 14 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    static constexpr int kPhaseUnroll       = ActiveTokens <= 4 ? 4 : 1;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, kPhaseUnroll,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// R2's longer K moves the stable four-to-sixteen-warp crossover to T=8.
// T=2..4: PhaseUnroll=4 is bit-exact and wins both cold and warm microbench
// (ninfer_nvfp4_mlp_down_t4_explore_v5).
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens <= 16 ? (ActiveTokens >= 8 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    static constexpr int kPhaseUnroll       = ActiveTokens <= 4 ? 4 : 1;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, kPhaseUnroll,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// MTP fc [5120,10240]: K sits between Residual6144 and Residual17408. Use TokenPacked +
// PhaseUnroll=4 across T=2..7. 16-warp at T≥8 remains the A16Only tail. AllowA4 W4A4
// for this geometry still starts at T=8 (M32N64 needs 8 live rows of the 32-wide tile).
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4MtpFcGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens <= 16 ? (ActiveTokens >= 8 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    static constexpr int kPhaseUnroll       = ActiveTokens <= 7 ? 4 : 1;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, kPhaseUnroll,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

} // namespace ninfer::ops::detail
