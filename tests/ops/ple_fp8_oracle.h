#pragma once

#include "ops/op_tester.h"
#include <cmath>
#include <cstdint>

namespace ninfer::test {

// Independent finite E4M3FN formula and represented BF16 tensor-scale boundary.
inline std::uint16_t fp8_ple_oracle(unsigned code,std::uint16_t scale) {
    const int exponent=(code>>3)&15,mantissa=code&7;
    const double magnitude=exponent==0?std::ldexp(double(mantissa),-9)
        :std::ldexp(1+double(mantissa)/8,exponent-7);
    const double value=std::copysign(magnitude,(code&128)?-1.0:1.0);
    return f32_to_bf16(float(value*bf16_to_f32(scale)));
}

} // namespace ninfer::test
