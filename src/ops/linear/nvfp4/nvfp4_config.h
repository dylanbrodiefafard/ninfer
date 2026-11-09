#pragma once

#include "ninfer/ops/a4_activation.h"
#include "ninfer/ops/dflash2_path_select.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// W4A4 activation scales are stored in [kNvfp4TmaBlockM tokens x kNvfp4ScaleTileGroups groups]
// tiles, 4 KiB each, contiguous and token-major, so one TMA request moves one tile slice and an MMA
// K256 stage reads one 16-byte run per token. Every W4A4 route reads this one layout.
inline constexpr std::int32_t kNvfp4TmaBlockM       = 256;
inline constexpr std::int32_t kNvfp4ScaleTileGroups = 16;
// The W4A4 workspace and the public A4 activation (ninfer/ops/a4_activation.h) share the plane.
static_assert(kNvfp4TmaBlockM == kA4ScaleTileTokens && kNvfp4ScaleTileGroups == kA4ScaleTileGroups);

// Token extent of the tiled plane: whole tiles. The quantizer zero-fills the padding scales of the
// last 16-token fragment, the only padding a consumer loads.
[[nodiscard]] constexpr std::int32_t nvfp4_w4a4_padded_tokens(std::int32_t tokens) noexcept {
    return ((tokens + kNvfp4TmaBlockM - 1) / kNvfp4TmaBlockM) * kNvfp4TmaBlockM;
}

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

using Nvfp4AttnInputGeometry      = Nvfp4GemvGeometry<14336, 5120>;
using Nvfp4GdnInputGeometry       = Nvfp4GemvGeometry<16384, 5120>;
using Nvfp4MlpGateUpGeometry      = Nvfp4GemvGeometry<34816, 5120>;
using Nvfp4Residual6144Geometry   = Nvfp4GemvGeometry<5120, 6144>;
using Nvfp4Residual17408Geometry  = Nvfp4GemvGeometry<5120, 17408>;
using Nvfp4DflashFeatureGeometry  = Nvfp4GemvGeometry<5120, 25600>;
using Nvfp4DflashQkvGeometry      = Nvfp4GemvGeometry<6144, 5120>;
using Nvfp4DflashAttnOutGeometry  = Nvfp4GemvGeometry<5120, 4096>;
using Nvfp4DflashConvProjGeometry = Nvfp4GemvGeometry<1280, 5120>;
using Nvfp4DflashSelectorGeometry = Nvfp4GemvGeometry<256, 5120>;
using Nvfp4MtpFcGeometry          = Nvfp4GemvGeometry<5120, 10240>;

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

// GEMM route of one W4A4 projection: an nvfp4_w4a4_mma_kernel schedule, named token tile x
// weight-row tile and K pipeline stages (the Nvfp4W4a4M* aliases in nvfp4_w4a4_mma.cuh), or
// the M256 TMA kernel.
enum class Nvfp4W4a4Route : std::uint8_t {
    M32N64S3,
    M32N64S4,
    M32N128,
    M64N64,
    M64N128S3,
    M128N128Pipelined,
    M128N128Resident,
    Tma,
};

// Per-shape W4A4 route by token count, shared by every projection Op over the shape. Each
// tier is the RTX 5090 cold-L2 public-Op winner, within 3%, of a sweep over T=2..1024 (steps
// of 2 through T=32, then 16). Both kernels launch the token tiles of one weight tile
// consecutively, so the weights stream from DRAM once and smaller token tiles buy CTAs for the
// narrow N=5120 projections without another weight pass. The TMA kernel's 256-token tiles win
// once the MMA tiles become compute-bound; between those, the cheapest token-tile count and
// wave fill decide. Examples, us: gate/up T=160 205 -> 94 and T=320 170 -> 137; 5120x17408
// T=256 125 -> 72. MtpFc (5120x10240) is unmeasured and takes the 5120x6144 table.
[[nodiscard]] constexpr Nvfp4W4a4Route nvfp4_w4a4_route(Nvfp4Problem problem, std::int32_t tokens) {
    using enum Nvfp4W4a4Route;
    switch (problem) {
    case Nvfp4Problem::MlpGateUp:
        if (tokens <= 64) { return M32N128; }
        if (tokens <= 128) { return M128N128Pipelined; }
        if (tokens <= 192) { return M64N128S3; }
        if (tokens <= 384) { return M128N128Pipelined; }
        return Tma;
    case Nvfp4Problem::GdnInput:
        if (tokens <= 32) { return M32N64S3; }
        if (tokens <= 64) { return M64N128S3; }
        if (tokens <= 128) { return M128N128Pipelined; }
        if (tokens <= 256) { return Tma; }
        if (tokens <= 320) { return M64N128S3; }
        if (tokens <= 512) { return Tma; }
        if (tokens <= 640) { return M128N128Resident; }
        return Tma;
    case Nvfp4Problem::AttnInput:
        if (tokens <= 32) { return M32N64S3; }
        if (tokens <= 96) { return M32N128; }
        if (tokens <= 128) { return M128N128Pipelined; }
        if (tokens <= 192) { return M64N128S3; }
        if (tokens <= 256) { return Tma; }
        if (tokens <= 384) { return M128N128Resident; }
        return Tma;
    case Nvfp4Problem::Residual17408:
        if (tokens <= 64) { return M32N64S4; }
        if (tokens <= 112) { return M32N64S3; }
        if (tokens <= 128) { return M64N64; }
        if (tokens <= 256) { return M64N128S3; }
        if (tokens <= 512) { return M128N128Pipelined; }
        return Tma;
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::MtpFc:
        if (tokens <= 64) { return M32N64S4; }
        if (tokens <= 112) { return M32N64S3; }
        if (tokens <= 256) { return M64N128S3; }
        if (tokens <= 512) { return M128N128Pipelined; }
        return Tma;
    case Nvfp4Problem::DflashFeature:
    case Nvfp4Problem::DflashQkv:
    case Nvfp4Problem::DflashAttnOut:
    case Nvfp4Problem::DflashConvProj:
    case Nvfp4Problem::DflashSelector:
        break;
    }
    throw std::invalid_argument("nvfp4 W4A4: DFlash2 problems are A16-only");
}

// Every W4A4 table routes all widths above this one to the TMA kernel.
inline constexpr std::int32_t kNvfp4W4a4LastMmaTokens = 640;

// Whether some width selects route for problem. Dispatchers instantiate only these MMA schedules.
[[nodiscard]] constexpr bool nvfp4_w4a4_route_reachable(Nvfp4Problem problem,
                                                        Nvfp4W4a4Route route) {
    for (std::int32_t tokens = 1; tokens <= kNvfp4W4a4LastMmaTokens + 1; ++tokens) {
        if (nvfp4_w4a4_route(problem, tokens) == route) { return true; }
    }
    return false;
}

static_assert(nvfp4_w4a4_route(Nvfp4Problem::AttnInput, kNvfp4W4a4LastMmaTokens + 1) ==
              Nvfp4W4a4Route::Tma);
static_assert(nvfp4_w4a4_route(Nvfp4Problem::GdnInput, kNvfp4W4a4LastMmaTokens + 1) ==
              Nvfp4W4a4Route::Tma);
static_assert(nvfp4_w4a4_route(Nvfp4Problem::MlpGateUp, kNvfp4W4a4LastMmaTokens + 1) ==
              Nvfp4W4a4Route::Tma);
static_assert(nvfp4_w4a4_route(Nvfp4Problem::Residual6144, kNvfp4W4a4LastMmaTokens + 1) ==
              Nvfp4W4a4Route::Tma);
static_assert(nvfp4_w4a4_route(Nvfp4Problem::Residual17408, kNvfp4W4a4LastMmaTokens + 1) ==
              Nvfp4W4a4Route::Tma);
static_assert(nvfp4_w4a4_route(Nvfp4Problem::MtpFc, kNvfp4W4a4LastMmaTokens + 1) ==
              Nvfp4W4a4Route::Tma);

// RTX 5090 SM count, the CTAs one wave holds at one resident CTA per SM.
inline constexpr std::int32_t kNvfp4Rtx5090SmCount = 170;

// The W4A8 MMA's wide path (N > 6144) compiles its eight-token panel count per launch; widths
// above kNvfp4W4a8WideMaxPanels panels repeat the 64-token tile.
inline constexpr int kNvfp4W4a8WideMaxPanels = 8;

[[nodiscard]] constexpr int nvfp4_w4a8_wide_panels(std::int32_t tokens) {
    const int panels = (tokens + 7) / 8;
    return panels < kNvfp4W4a8WideMaxPanels ? panels : kNvfp4W4a8WideMaxPanels;
}

// Token tiles of the BF16-activation (A16) MMA for the DFlash context projections above 48 tokens,
// which run once per prefill chunk: the feature projection [5120, 25600] and the context K/V
// projection (the K and V row blocks of [6144, 5120], 2048 rows). Narrow keeps the narrow
// family's own M16/M32/M48 token tile; the others place 64 tokens on MMA M and BlockN output rows
// on MMA N.
enum class Nvfp4A16PrefillTile : std::uint8_t { Narrow, M64N32, M64N64, M64N128 };

// Every CTA streams the whole K panel, and measured time steps once per 170 CTAs (one per RTX 5090
// SM), so each width picks the tile whose CTA count fills those waves best. The tables are the
// per-width winners measured on the RTX 5090 with cold L2 (ninfer_linear_bench,
// ninfer_linear_kv_projection_bench).
inline constexpr std::int32_t kNvfp4DflashFeatureLastNarrowTokens = 48;

[[nodiscard]] constexpr Nvfp4A16PrefillTile nvfp4_dflash_feature_a16_tile(std::int32_t tokens) {
    if (tokens <= kNvfp4DflashFeatureLastNarrowTokens) { return Nvfp4A16PrefillTile::Narrow; }
    if (tokens <= 64) { return Nvfp4A16PrefillTile::M64N32; }
    if (tokens <= 128) { return Nvfp4A16PrefillTile::M64N64; }
    if (tokens <= 256) { return Nvfp4A16PrefillTile::M64N128; }
    if (tokens <= 384) { return Nvfp4A16PrefillTile::M64N64; }
    if (tokens <= 512) { return Nvfp4A16PrefillTile::M64N128; }
    if (tokens <= 640) { return Nvfp4A16PrefillTile::M64N64; }
    return Nvfp4A16PrefillTile::M64N128;
}

// The feature table keeps only widths through kNvfp4DflashFeatureLastNarrowTokens on the narrow
// family, so the feature projection never instantiates the narrow family's 64-token tile. The
// table is constant above its last breakpoint, which this check covers.
[[nodiscard]] constexpr bool nvfp4_dflash_feature_narrow_only_through_last() {
    for (std::int32_t tokens = kNvfp4DflashFeatureLastNarrowTokens + 1; tokens <= 1024; ++tokens) {
        if (nvfp4_dflash_feature_a16_tile(tokens) == Nvfp4A16PrefillTile::Narrow) { return false; }
    }
    return true;
}

static_assert(nvfp4_dflash_feature_narrow_only_through_last());

[[nodiscard]] constexpr Nvfp4A16PrefillTile nvfp4_dflash_kv_a16_tile(std::int32_t tokens) {
    if (tokens <= 64) { return Nvfp4A16PrefillTile::Narrow; }
    if (tokens <= 128) { return Nvfp4A16PrefillTile::M64N32; }
    if (tokens <= 192) { return Nvfp4A16PrefillTile::Narrow; }
    if (tokens <= 320) { return Nvfp4A16PrefillTile::M64N64; }
    if (tokens <= 448) { return Nvfp4A16PrefillTile::Narrow; }
    if (tokens <= 640) { return Nvfp4A16PrefillTile::M64N128; }
    if (tokens <= 704) { return Nvfp4A16PrefillTile::Narrow; }
    if (tokens <= 960) { return Nvfp4A16PrefillTile::M64N64; }
    if (tokens <= 1280) { return Nvfp4A16PrefillTile::M64N128; }
    return Nvfp4A16PrefillTile::M64N64;
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
inline constexpr std::int32_t kNvfp4FirstW4a4AttnInput     = kA4AttnInputMinTokens;
inline constexpr std::int32_t kNvfp4FirstW4a4GdnInput      = kA4GdnInputMinTokens;
inline constexpr std::int32_t kNvfp4FirstW4a4MlpGateUp     = kA4MlpGateUpMinTokens;
inline constexpr std::int32_t kNvfp4FirstW4a4Residual6144  = kA4Residual6144MinTokens;
inline constexpr std::int32_t kNvfp4FirstW4a4Residual17408 = kA4Residual17408MinTokens;
inline constexpr std::int32_t kNvfp4FirstW4a4MtpFc         = 8;
// AllowA8 is the verification policy: every verify width (W>=2) and aggregate uses W4A8, so a
// request's arithmetic does not depend on its draft width or batch. T=1 stays on the A16 GEMV.
inline constexpr std::int32_t kNvfp4FirstA8 = 2;

// Generic NVFP4 Linear serves the DFlash2 drafter (whose packed proposal launches span at most
// kNvfp4DflashMaxProposalTokens columns) and MTP fc; Text projections run through their fused
// Ops. The predicates below are the token counts those callers launch, and dispatchers
// instantiate only kernels for them.
inline constexpr std::int32_t kNvfp4DflashMaxProposalTokens =
    kDflash2MaxChainWidth * kDflash2MaxBatch;
// The DFlash conv projection packs proposal blocks into passes of at most this many columns, the
// widest SmallT instance, so every pass keeps the lane reduction of its per-request panels.
inline constexpr std::int32_t kNvfp4DflashConvProjPassTokens = kNvfp4LastSmallT;

// Whether some packed DFlash conv projection pass has T columns: m blocks of W = k + 1 columns,
// with m at most the batch bound and the blocks that fit one pass.
[[nodiscard]] constexpr bool nvfp4_dflash_conv_proj_pass_tokens(std::int32_t tokens) {
    for (std::int32_t width = 2; width <= kDflash2MaxChainWidth; ++width) {
        const std::int32_t max_blocks = kDflash2MaxBatch < kNvfp4DflashConvProjPassTokens / width
                                            ? kDflash2MaxBatch
                                            : kNvfp4DflashConvProjPassTokens / width;
        if (tokens % width == 0 && tokens / width <= max_blocks) { return true; }
    }
    return false;
}

// Token counts in [kNvfp4FirstSmallT, kNvfp4LastSmallT] at which generic Linear launches the A16
// SmallT kernel: the DFlash MLP down at the k=1 proposal width (wider blocks take W4A4), the conv
// projection passes, the selector projection at T=k, and MTP fc. The DFlash feature, QKV and
// attention-output projections run the A16 MMA at T>=2 instead.
[[nodiscard]] constexpr bool nvfp4_linear_small_t_admitted(Nvfp4Problem problem,
                                                           std::int32_t tokens) {
    if (tokens < kNvfp4FirstSmallT || tokens > kNvfp4LastSmallT) { return false; }
    switch (problem) {
    case Nvfp4Problem::Residual17408:
        return tokens < kNvfp4FirstW4a4Residual17408;
    case Nvfp4Problem::DflashConvProj:
        return nvfp4_dflash_conv_proj_pass_tokens(tokens);
    case Nvfp4Problem::DflashSelector:
        return tokens <= kDflash2MaxDraftTokens;
    case Nvfp4Problem::MtpFc:
        return true;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::DflashFeature:
    case Nvfp4Problem::DflashQkv:
    case Nvfp4Problem::DflashAttnOut:
        return false;
    }
    return false;
}

// Token counts at which generic Linear under AllowA4 launches W4A4: the DFlash MLP gate/up and
// down from their cutovers through the widest packed proposal, and MTP fc from its cutover.
[[nodiscard]] constexpr bool nvfp4_linear_w4a4_admitted(Nvfp4Problem problem, std::int32_t tokens) {
    switch (problem) {
    case Nvfp4Problem::MlpGateUp:
        return tokens >= kNvfp4FirstW4a4MlpGateUp && tokens <= kNvfp4DflashMaxProposalTokens;
    case Nvfp4Problem::Residual17408:
        return tokens >= kNvfp4FirstW4a4Residual17408 && tokens <= kNvfp4DflashMaxProposalTokens;
    case Nvfp4Problem::MtpFc:
        return tokens >= kNvfp4FirstW4a4MtpFc;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::DflashFeature:
    case Nvfp4Problem::DflashQkv:
    case Nvfp4Problem::DflashAttnOut:
    case Nvfp4Problem::DflashConvProj:
    case Nvfp4Problem::DflashSelector:
        return false;
    }
    return false;
}

// Whether some width admitted by generic Linear selects route for problem. The generic W4A4
// dispatcher instantiates only these schedules.
[[nodiscard]] constexpr bool nvfp4_linear_w4a4_route_reachable(Nvfp4Problem problem,
                                                               Nvfp4W4a4Route route) {
    for (std::int32_t tokens = 1; tokens <= kNvfp4W4a4LastMmaTokens + 1; ++tokens) {
        if (nvfp4_linear_w4a4_admitted(problem, tokens) &&
            nvfp4_w4a4_route(problem, tokens) == route) {
            return true;
        }
    }
    return false;
}

static_assert(nvfp4_linear_w4a4_route_reachable(Nvfp4Problem::MlpGateUp, Nvfp4W4a4Route::M32N128));
static_assert(!nvfp4_linear_w4a4_route_reachable(Nvfp4Problem::MlpGateUp, Nvfp4W4a4Route::Tma));

// RTX 5090 cold-cache winner for contiguous Linear output at the DFlash selector widths
// (T=2..kDflash2MaxDraftTokens). T=2..4 amortizes activation loads through shared staging; wider
// panels keep one packed activation tile per warp.
template <class Geometry, int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kDflash2MaxDraftTokens);
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type = Nvfp4SmallTSchedule<8, 1, 2, 16, ActiveTokens, 1, kActivationAccess,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// The DFlash conv projection packs proposal blocks into passes of up to
// kNvfp4DflashConvProjPassTokens columns, so every T keeps the 16-value lane reduction of its
// per-request panels. The warp-count changes are measured occupancy/register crossovers, not
// semantic frontiers.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4DflashConvProjGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4DflashConvProjPassTokens);
    static constexpr int kWarpsPerCta = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 13 ? 16 : 8);
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type = Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, 16, ActiveTokens, 1, kActivationAccess,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// The A16 GDN input projection runs only below its W4A4 cutover. It keeps the T=1 GDN GEMV's
// 8-warp StagedRaw reduction with four accumulator chains and stages its activation columns in
// shared memory.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens < kNvfp4FirstW4a4GdnInput);
    using Type =
        Nvfp4SmallTSchedule<8, 1, 2, 16, ActiveTokens, 4, Nvfp4SmallTActivationAccess::SharedPhase,
                            Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// The residual projections run A16 SmallT only below their W4A4 cutovers (T<=4 at K=6144, T=2 at
// K=17408), where four-warp CTAs win at N=5120. PhaseUnroll=4 is bit-exact and wins cold on the
// RTX 5090 at both K.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens < kNvfp4FirstW4a4Residual6144);
    using Type =
        Nvfp4SmallTSchedule<4, 1, 2, 16, ActiveTokens, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens < kNvfp4FirstW4a4Residual17408);
    using Type =
        Nvfp4SmallTSchedule<4, 1, 2, 16, ActiveTokens, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
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
