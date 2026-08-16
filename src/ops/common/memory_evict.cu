#include "ops/common/memory.cuh"

namespace ninfer::ops {

__device__ __noinline__ void cp_async_evict_first_16_noinline(void* smem_dst,
                                                              const void* gmem_src) {
    const unsigned long long pol = l2_evict_first_policy();
    cp_async_evict_first_16(smem_dst, gmem_src, pol);
}

}  // namespace ninfer::ops
