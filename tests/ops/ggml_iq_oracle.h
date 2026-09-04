#pragma once

#include <array>
#include <stdexcept>

namespace ninfer::test::iq_oracle {

// Test-owned format witnesses.  These values are independently embedded from the GGML format
// definition; this header must never include or alias the production CUDA codebook data.
inline constexpr std::array<int, 16> kIq4Nl = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

inline constexpr std::array<int, 8> kIq1Grid0 = {-1, -1, -1, -1, -1, -1, -1, -1};
inline constexpr std::array<int, 8> kIq1Grid1 = {1, -1, -1, -1, -1, -1, -1, -1};
inline constexpr std::array<int, 8> kIq1Grid2 = {0, 0, -1, -1, -1, -1, -1, -1};
inline constexpr std::array<int, 8> kIq1Grid2047 = {1, 1, 1, 1, 1, 1, 1, 1};

inline const std::array<int, 8>& iq1_grid(int index) {
    switch (index) {
    case 0:
        return kIq1Grid0;
    case 1:
        return kIq1Grid1;
    case 2:
        return kIq1Grid2;
    case 2047:
        return kIq1Grid2047;
    default:
        throw std::logic_error("IQ1_S test fixture used a non-witness grid index");
    }
}

inline constexpr std::array<int, 8> kIq2Grid0 = {8, 8, 8, 8, 8, 8, 8, 8};
inline constexpr std::array<int, 8> kIq2Grid1 = {43, 8, 8, 8, 8, 8, 8, 8};
inline constexpr std::array<int, 8> kIq2Grid2 = {25, 25, 8, 8, 8, 8, 8, 8};
inline constexpr std::array<int, 8> kIq2Grid255 = {8, 25, 8, 8, 25, 43, 43, 43};

inline const std::array<int, 8>& iq2_grid(int index) {
    switch (index) {
    case 0:
        return kIq2Grid0;
    case 1:
        return kIq2Grid1;
    case 2:
        return kIq2Grid2;
    case 255:
        return kIq2Grid255;
    default:
        throw std::logic_error("IQ2_XXS test fixture used a non-witness grid index");
    }
}

} // namespace ninfer::test::iq_oracle
