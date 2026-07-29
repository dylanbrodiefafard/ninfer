#pragma once

#include "ops/linear/bf16/bf16_gemm_mma.cuh"

namespace ninfer::ops::detail {

// Measured large-T production schedule for the [14336,5120] BF16 computation core. Keep the
// schedule as a type so tuning can replace any tile, pipeline, cache, raster, or fragment choice
// without changing either Linear or semantic-Op dispatch.
using Bf16MmaProductionSchedule =
    Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

} // namespace ninfer::ops::detail
