#pragma once

#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"

namespace ninfer::ops::detail {

template <class Geometry>
struct Fp8LinearA8ProductionSchedule;

using Fp8LinearA8DefaultSchedule =
    Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                   Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;

template <>
struct Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

#define NINFER_FP8_A8_DEFAULT_SCHEDULE(Geometry)                                                    \
    template <>                                                                                     \
    struct Fp8LinearA8ProductionSchedule<Geometry> {                                                \
        using Type = Fp8LinearA8DefaultSchedule;                                                    \
    }

NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows10240K2560Geometry);
NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows6144K2560Geometry);
NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows12288K2560Geometry);
NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows512K2560Geometry);
NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows2560K6144Geometry);
NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows640K2560Geometry);
NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows1280K2560Geometry);
NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows2560K640Geometry);
NINFER_FP8_A8_DEFAULT_SCHEDULE(Fp8Rows2560K2560Geometry);

#undef NINFER_FP8_A8_DEFAULT_SCHEDULE

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Rows320K10240Geometry> {
    using Type = Fp8MmaSchedule<64, 64, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Rows10240K320Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 64, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

} // namespace ninfer::ops::detail
