#include "targets/qwen4/real_oracle_common.h"

#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/qwen4_sparse_moe.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;
namespace ops = ninfer::ops;
using ninfer::DType;
using ninfer::DeviceBuffer;
using ninfer::PinnedHostBuffer;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::WorkspaceArena;
using ninfer::test::GuardedDeviceBuffer;
using ninfer::test::PointwiseCriterion;
using ninfer::test::ReductionCriterion;
using ninfer::test::bf16_to_f32;
using ninfer::test::from_device;
using ninfer::test::from_device_bf16;
using ninfer::test::to_device;
using ninfer::test::verify_pointwise;
using ninfer::test::verify_reduction;
using namespace ninfer::test::qwen4::real_oracle;

namespace {

constexpr std::int32_t kHidden = ops::kQwen4SparseMoeHidden;
constexpr std::int32_t kBranches = 4;
constexpr std::int32_t kFlat = kBranches * kHidden;
constexpr std::int32_t kGrRank = 320;
constexpr std::int32_t kExperts = ops::kQwen4SparseMoeExperts;
constexpr std::int32_t kTopK = ops::kQwen4SparseMoeTopK;
constexpr std::int32_t kIntermediate = ops::kQwen4SparseMoeIntermediate;
constexpr std::size_t kIq1BlockValues = 256;
constexpr std::size_t kIq1BlockBytes = 50;
constexpr std::size_t kIq4BlockValues = 32;
constexpr std::size_t kIq4BlockBytes = 18;
constexpr std::int32_t kAccumulatedPosition = 221;
constexpr std::size_t kQ8DownBytes =
    static_cast<std::size_t>(kGrRank) * kFlat / kQ8BlockValues * kQ8BlockBytes;
constexpr std::size_t kQ8UpBytes =
    static_cast<std::size_t>(kFlat) * kGrRank / kQ8BlockValues * kQ8BlockBytes;

// Pinned llama_tokenize output for one paragraph including its terminal LF. The 601-token
// evidence repeats this paragraph; position 221 is offset 49 in its third repetition.
constexpr std::array<std::int32_t, 86> kFrozenParagraph = {
    48, 16451, 17120, 22188, 11988, 3817, 19039, 888, 264, 2716, 8097, 40701, 13, 561,
    1558, 15339, 1754, 3299, 303, 1906, 321, 54004, 1092, 3905, 1727, 13, 3931, 921,
    13224, 20480, 16338, 1528, 11, 6326, 13224, 62586, 6575, 2193, 11, 321, 32335,
    11312, 5000, 3955, 10885, 13, 1061, 14648, 13901, 5533, 13983, 19464, 12, 23,
    1414, 11, 14733, 59429, 11, 321, 3213, 10885, 364, 799, 2526, 10756, 14751,
    1931, 19221, 3136, 13, 11116, 7193, 369, 33625, 17066, 5721, 11, 524, 264,
    3591, 883, 3992, 4131, 13, 198,
};
static_assert(kFrozenParagraph[kAccumulatedPosition % kFrozenParagraph.size()] == 5533);
static_assert(kFrozenParagraph[(kAccumulatedPosition + 1) % kFrozenParagraph.size()] == 13983);

// These are the established complete-Op criteria from test_qwen4_sparse_moe.cpp. This cell
// changes only the represented x and artifact weights.
constexpr ReductionCriterion kOutputCriterion{/*relative_l2=*/2.5 / 255.0,
                                               /*gross_absolute=*/1.0 / 32768.0,
                                               /*gross_relative_to_max_reference=*/2.0 / 255.0};
constexpr PointwiseCriterion kRouteWeightCriterion{
    /*absolute=*/128.0 * std::numeric_limits<float>::epsilon(),
    /*relative=*/128.0 * std::numeric_limits<float>::epsilon(),
};
constexpr ReductionCriterion kGrReadCriterion{/*relative_l2=*/6.0e-3,
                                               /*gross_absolute=*/4.0e-3,
                                               /*gross_relative_to_max_reference=*/5.0e-3};
constexpr ReductionCriterion kGrScaleCriterion{/*relative_l2=*/3.5e-3,
                                                /*gross_absolute=*/1.5e-3,
                                                /*gross_relative_to_max_reference=*/3.0e-3};
constexpr ReductionCriterion kGrInjectCriterion{/*relative_l2=*/3.0e-3,
                                                 /*gross_absolute=*/2.0e-3,
                                                 /*gross_relative_to_max_reference=*/2.0e-3};

constexpr std::array<int, 16> kIq4Nl = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// Test-owned IQ1_S format table. Each little-endian-equivalent word carries eight 2-bit ternary
// digits, least-significant digit first (0 -> -1, 1 -> 0, 2 -> +1). It is independently embedded
// from the pinned GGML format data and does not include or alias the production CUDA table.
constexpr std::array<std::uint16_t, 2048> kIq1Packed = {
    0x0000U, 0x0002U, 0x0005U, 0x0008U, 0x000aU, 0x0011U, 0x0015U, 0x0020U, 0x0022U, 0x0028U, 0x002aU, 0x0045U,
    0x0051U, 0x0054U, 0x0056U, 0x0065U, 0x0080U, 0x0082U, 0x0088U, 0x008aU, 0x0095U, 0x00a0U, 0x00a2U, 0x00a8U,
    0x00aaU, 0x0104U, 0x0105U, 0x0111U, 0x0114U, 0x0116U, 0x0119U, 0x011aU, 0x0125U, 0x0141U, 0x0146U, 0x0149U,
    0x0152U, 0x0155U, 0x015aU, 0x0161U, 0x0164U, 0x0166U, 0x0168U, 0x0185U, 0x0191U, 0x0194U, 0x0196U, 0x01a5U,
    0x0200U, 0x0202U, 0x0208U, 0x020aU, 0x0215U, 0x0220U, 0x0222U, 0x0228U, 0x022aU, 0x0245U, 0x0251U, 0x0259U,
    0x0264U, 0x0269U, 0x0280U, 0x0282U, 0x0288U, 0x028aU, 0x0291U, 0x0295U, 0x0299U, 0x02a0U, 0x02a2U, 0x02a8U,
    0x02aaU, 0x0411U, 0x0414U, 0x0416U, 0x0425U, 0x0441U, 0x0449U, 0x0455U, 0x045aU, 0x0464U, 0x0465U, 0x0491U,
    0x0499U, 0x04a5U, 0x0501U, 0x0504U, 0x0505U, 0x0506U, 0x0515U, 0x0518U, 0x051aU, 0x0529U, 0x0540U, 0x0545U,
    0x054aU, 0x0550U, 0x0551U, 0x0554U, 0x0555U, 0x0556U, 0x0559U, 0x0560U, 0x0562U, 0x0565U, 0x0568U, 0x056aU,
    0x0581U, 0x0591U, 0x0595U, 0x0598U, 0x059aU, 0x05a1U, 0x05a4U, 0x05a5U, 0x05a6U, 0x05a9U, 0x0614U, 0x0619U,
    0x0641U, 0x0644U, 0x0650U, 0x0652U, 0x0655U, 0x0658U, 0x0660U, 0x0661U, 0x0666U, 0x0669U, 0x0685U, 0x0691U,
    0x0694U, 0x0699U, 0x0800U, 0x0802U, 0x0808U, 0x080aU, 0x0815U, 0x0820U, 0x0822U, 0x0828U, 0x082aU, 0x0845U,
    0x0851U, 0x0856U, 0x0865U, 0x0880U, 0x0882U, 0x0888U, 0x088aU, 0x0895U, 0x08a0U, 0x08a2U, 0x08a8U, 0x08aaU,
    0x0905U, 0x0911U, 0x0914U, 0x0919U, 0x0924U, 0x0925U, 0x0941U, 0x0950U, 0x0951U, 0x0955U, 0x0961U, 0x0964U,
    0x0969U, 0x0991U, 0x0994U, 0x0996U, 0x0999U, 0x09a5U, 0x0a00U, 0x0a02U, 0x0a08U, 0x0a0aU, 0x0a15U, 0x0a20U,
    0x0a22U, 0x0a28U, 0x0a2aU, 0x0a45U, 0x0a51U, 0x0a59U, 0x0a61U, 0x0a65U, 0x0a80U, 0x0a82U, 0x0a85U, 0x0a88U,
    0x0a8aU, 0x0a95U, 0x0aa0U, 0x0aa2U, 0x0aa8U, 0x0aaaU, 0x1010U, 0x1011U, 0x1014U, 0x1019U, 0x1024U, 0x1025U,
    0x1041U, 0x1044U, 0x1050U, 0x1055U, 0x1058U, 0x1061U, 0x1064U, 0x1065U, 0x1069U, 0x1091U, 0x1094U, 0x1096U,
    0x10a1U, 0x10a5U, 0x1101U, 0x1104U, 0x1106U, 0x1109U, 0x1110U, 0x1112U, 0x1115U, 0x1118U, 0x1121U, 0x1124U,
    0x1129U, 0x1145U, 0x114aU, 0x1150U, 0x1151U, 0x1152U, 0x1154U, 0x1155U, 0x1156U, 0x1159U, 0x1160U, 0x1165U,
    0x1184U, 0x1192U, 0x1195U, 0x11a1U, 0x11a4U, 0x1211U, 0x1214U, 0x1216U, 0x1225U, 0x1240U, 0x1246U, 0x1249U,
    0x1252U, 0x1255U, 0x1258U, 0x125aU, 0x1264U, 0x1266U, 0x1285U, 0x1291U, 0x1294U, 0x1296U, 0x12a5U, 0x1401U,
    0x1406U, 0x1409U, 0x1414U, 0x1415U, 0x1418U, 0x1419U, 0x1421U, 0x1426U, 0x1441U, 0x1445U, 0x1446U, 0x1448U,
    0x144aU, 0x1451U, 0x1454U, 0x1455U, 0x1456U, 0x1459U, 0x1462U, 0x1465U, 0x1468U, 0x1484U, 0x1489U, 0x1490U,
    0x1494U, 0x1495U, 0x1498U, 0x1499U, 0x149aU, 0x14a1U, 0x14a4U, 0x14a5U, 0x14a9U, 0x1502U, 0x1505U, 0x150aU,
    0x1511U, 0x1514U, 0x1515U, 0x1516U, 0x1519U, 0x1520U, 0x1522U, 0x1525U, 0x1528U, 0x152aU, 0x1541U, 0x1544U,
    0x1545U, 0x1546U, 0x1551U, 0x1552U, 0x1554U, 0x1555U, 0x1556U, 0x1559U, 0x155aU, 0x1561U, 0x1564U, 0x1565U,
    0x1566U, 0x1569U, 0x1580U, 0x1582U, 0x1584U, 0x1585U, 0x1588U, 0x158aU, 0x1590U, 0x1591U, 0x1594U, 0x1595U,
    0x1596U, 0x1599U, 0x159aU, 0x15a0U, 0x15a2U, 0x15a5U, 0x1601U, 0x1604U, 0x1605U, 0x1606U, 0x1615U, 0x1616U,
    0x1618U, 0x161aU, 0x1621U, 0x1626U, 0x1640U, 0x1642U, 0x1644U, 0x1645U, 0x1648U, 0x164aU, 0x1651U, 0x1655U,
    0x1656U, 0x1658U, 0x1659U, 0x1661U, 0x1664U, 0x1665U, 0x1668U, 0x1669U, 0x166aU, 0x1686U, 0x168aU, 0x1692U,
    0x1695U, 0x16a4U, 0x16a9U, 0x1811U, 0x1816U, 0x1825U, 0x1841U, 0x1844U, 0x1846U, 0x1849U, 0x1850U, 0x1855U,
    0x1858U, 0x185aU, 0x1860U, 0x1861U, 0x1864U, 0x1866U, 0x1869U, 0x1885U, 0x1891U, 0x1894U, 0x18a5U, 0x1910U,
    0x1912U, 0x1915U, 0x191aU, 0x1921U, 0x1925U, 0x1942U, 0x1944U, 0x1945U, 0x1948U, 0x1951U, 0x1954U, 0x1955U,
    0x1956U, 0x1959U, 0x195aU, 0x1960U, 0x1965U, 0x196aU, 0x1989U, 0x1991U, 0x1992U, 0x1995U, 0x1998U, 0x19a1U,
    0x19a6U, 0x19a9U, 0x1a09U, 0x1a16U, 0x1a24U, 0x1a26U, 0x1a44U, 0x1a46U, 0x1a49U, 0x1a50U, 0x1a52U, 0x1a55U,
    0x1a58U, 0x1a61U, 0x1a66U, 0x1a69U, 0x1a85U, 0x1a91U, 0x1a96U, 0x1a9aU, 0x2000U, 0x2002U, 0x2008U, 0x200aU,
    0x2015U, 0x2020U, 0x2022U, 0x2025U, 0x2028U, 0x202aU, 0x2045U, 0x2051U, 0x2059U, 0x2061U, 0x2065U, 0x2080U,
    0x2082U, 0x2088U, 0x208aU, 0x2095U, 0x20a0U, 0x20a2U, 0x20a5U, 0x20a8U, 0x20aaU, 0x2105U, 0x2111U, 0x2114U,
    0x2119U, 0x2125U, 0x2142U, 0x2144U, 0x2149U, 0x2155U, 0x2158U, 0x215aU, 0x2161U, 0x2164U, 0x2165U, 0x2166U,
    0x2185U, 0x2190U, 0x2196U, 0x2199U, 0x21a5U, 0x2201U, 0x2208U, 0x220aU, 0x2211U, 0x2215U, 0x2220U, 0x2222U,
    0x2228U, 0x222aU, 0x2245U, 0x2251U, 0x2256U, 0x2259U, 0x2265U, 0x2281U, 0x2288U, 0x228aU, 0x2291U, 0x2295U,
    0x22a0U, 0x22a2U, 0x22a8U, 0x22aaU, 0x2405U, 0x2414U, 0x2416U, 0x2419U, 0x2425U, 0x2444U, 0x2445U, 0x2446U,
    0x2449U, 0x2452U, 0x2455U, 0x2458U, 0x245aU, 0x2466U, 0x2485U, 0x2491U, 0x2494U, 0x2499U, 0x24a1U, 0x24a5U,
    0x2509U, 0x2515U, 0x2521U, 0x2529U, 0x2540U, 0x2545U, 0x2548U, 0x2551U, 0x2554U, 0x2555U, 0x2559U, 0x2562U,
    0x2565U, 0x2568U, 0x2589U, 0x2590U, 0x2594U, 0x2595U, 0x2598U, 0x259aU, 0x25a1U, 0x25a4U, 0x25a6U, 0x25a9U,
    0x2605U, 0x2610U, 0x2612U, 0x2619U, 0x2625U, 0x2641U, 0x2649U, 0x2655U, 0x2660U, 0x2661U, 0x2669U, 0x2684U,
    0x2686U, 0x2690U, 0x269aU, 0x2800U, 0x2802U, 0x2808U, 0x280aU, 0x2815U, 0x2820U, 0x2822U, 0x2828U, 0x282aU,
    0x2845U, 0x2851U, 0x2854U, 0x2865U, 0x2880U, 0x2882U, 0x2888U, 0x288aU, 0x28a0U, 0x28a2U, 0x28a8U, 0x28aaU,
    0x2909U, 0x2911U, 0x2914U, 0x2919U, 0x2925U, 0x2946U, 0x2949U, 0x2952U, 0x2955U, 0x2961U, 0x2964U, 0x2966U,
    0x2969U, 0x2985U, 0x2990U, 0x2996U, 0x2999U, 0x29a4U, 0x29a5U, 0x2a00U, 0x2a02U, 0x2a08U, 0x2a0aU, 0x2a20U,
    0x2a22U, 0x2a28U, 0x2a2aU, 0x2a45U, 0x2a51U, 0x2a56U, 0x2a59U, 0x2a65U, 0x2a80U, 0x2a82U, 0x2a88U, 0x2a8aU,
    0x2a95U, 0x2aa0U, 0x2aa2U, 0x2aa8U, 0x2aaaU, 0x4005U, 0x4011U, 0x4016U, 0x4025U, 0x4049U, 0x4052U, 0x4055U,
    0x4058U, 0x405aU, 0x4061U, 0x4064U, 0x4066U, 0x4094U, 0x4099U, 0x40a1U, 0x40a6U, 0x4100U, 0x4101U, 0x4104U,
    0x4106U, 0x4109U, 0x4112U, 0x4115U, 0x4116U, 0x4118U, 0x411aU, 0x4121U, 0x4126U, 0x4129U, 0x4145U, 0x4148U,
    0x414aU, 0x4151U, 0x4154U, 0x4155U, 0x4156U, 0x4159U, 0x415aU, 0x4165U, 0x4168U, 0x416aU, 0x4181U, 0x4184U,
    0x4186U, 0x4190U, 0x4192U, 0x4195U, 0x41a0U, 0x41a1U, 0x41a2U, 0x4205U, 0x4211U, 0x4214U, 0x4216U, 0x4225U,
    0x4241U, 0x4252U, 0x4255U, 0x425aU, 0x4264U, 0x4269U, 0x4289U, 0x4294U, 0x42a5U, 0x4401U, 0x4415U, 0x4419U,
    0x4429U, 0x4445U, 0x4448U, 0x444aU, 0x4451U, 0x4454U, 0x4455U, 0x4456U, 0x4461U, 0x4462U, 0x4465U, 0x4468U,
    0x446aU, 0x4481U, 0x4486U, 0x4489U, 0x4490U, 0x4492U, 0x4495U, 0x44a0U, 0x44a1U, 0x44a9U, 0x4501U, 0x4502U,
    0x4505U, 0x450aU, 0x4511U, 0x4514U, 0x4515U, 0x4516U, 0x4519U, 0x4520U, 0x4525U, 0x452aU, 0x4541U, 0x4544U,
    0x4545U, 0x4546U, 0x4549U, 0x4550U, 0x4551U, 0x4554U, 0x4555U, 0x4556U, 0x4558U, 0x4559U, 0x4561U, 0x4564U,
    0x4565U, 0x4566U, 0x4569U, 0x4582U, 0x4584U, 0x4585U, 0x4588U, 0x4591U, 0x4594U, 0x4595U, 0x4596U, 0x4599U,
    0x459aU, 0x45a5U, 0x45a8U, 0x45aaU, 0x4601U, 0x4605U, 0x4609U, 0x4614U, 0x4615U, 0x4618U, 0x461aU, 0x4621U,
    0x4624U, 0x4629U, 0x4640U, 0x4642U, 0x4645U, 0x4648U, 0x4650U, 0x4651U, 0x4652U, 0x4655U, 0x4656U, 0x4659U,
    0x4662U, 0x4665U, 0x4668U, 0x4681U, 0x4685U, 0x468aU, 0x4694U, 0x4695U, 0x46a1U, 0x46a4U, 0x46a6U, 0x4805U,
    0x4811U, 0x4815U, 0x481aU, 0x4825U, 0x4842U, 0x4849U, 0x4850U, 0x4855U, 0x4858U, 0x4861U, 0x4864U, 0x4866U,
    0x4869U, 0x4885U, 0x4891U, 0x4894U, 0x4896U, 0x4899U, 0x48a5U, 0x4901U, 0x4905U, 0x4906U, 0x490aU, 0x4910U,
    0x4914U, 0x4915U, 0x4918U, 0x4921U, 0x4924U, 0x4926U, 0x4940U, 0x4945U, 0x494aU, 0x4951U, 0x4952U, 0x4954U,
    0x4955U, 0x4956U, 0x4959U, 0x4960U, 0x4962U, 0x4965U, 0x4966U, 0x496aU, 0x4986U, 0x4989U, 0x4992U, 0x4995U,
    0x4996U, 0x4998U, 0x49a1U, 0x49a4U, 0x49a6U, 0x49a9U, 0x4a16U, 0x4a44U, 0x4a46U, 0x4a49U, 0x4a55U, 0x4a58U,
    0x4a5aU, 0x4a64U, 0x4a69U, 0x4a94U, 0x4aa5U, 0x5001U, 0x5004U, 0x5005U, 0x5006U, 0x5009U, 0x5012U, 0x5015U,
    0x501aU, 0x5021U, 0x5024U, 0x5029U, 0x5040U, 0x5045U, 0x5048U, 0x5051U, 0x5054U, 0x5055U, 0x5056U, 0x5059U,
    0x5065U, 0x5068U, 0x5086U, 0x5089U, 0x5095U, 0x5098U, 0x50a0U, 0x50a1U, 0x50a6U, 0x50a9U, 0x5105U, 0x5108U,
    0x5109U, 0x510aU, 0x5111U, 0x5114U, 0x5115U, 0x5116U, 0x5118U, 0x5119U, 0x5120U, 0x5125U, 0x5126U, 0x5128U,
    0x512aU, 0x5141U, 0x5144U, 0x5145U, 0x5146U, 0x5149U, 0x5150U, 0x5151U, 0x5152U, 0x5154U, 0x5155U, 0x5156U,
    0x5158U, 0x5159U, 0x515aU, 0x5161U, 0x5164U, 0x5165U, 0x5166U, 0x5169U, 0x5182U, 0x5185U, 0x5191U, 0x5194U,
    0x5195U, 0x5196U, 0x5199U, 0x51a0U, 0x51a5U, 0x51aaU, 0x5201U, 0x5206U, 0x5212U, 0x5215U, 0x521aU, 0x5221U,
    0x5224U, 0x5242U, 0x5245U, 0x524aU, 0x5251U, 0x5254U, 0x5255U, 0x5256U, 0x5259U, 0x5262U, 0x5265U, 0x5285U,
    0x5290U, 0x5292U, 0x5295U, 0x5299U, 0x529aU, 0x52a4U, 0x5404U, 0x5405U, 0x5411U, 0x5414U, 0x5415U, 0x5416U,
    0x5418U, 0x5419U, 0x5421U, 0x5425U, 0x5428U, 0x542aU, 0x5441U, 0x5444U, 0x5445U, 0x5446U, 0x5449U, 0x544aU,
    0x5450U, 0x5451U, 0x5454U, 0x5455U, 0x5456U, 0x5458U, 0x5459U, 0x545aU, 0x5461U, 0x5462U, 0x5464U, 0x5465U,
    0x5466U, 0x5469U, 0x5480U, 0x5488U, 0x548aU, 0x5491U, 0x5494U, 0x5495U, 0x5496U, 0x5499U, 0x54a1U, 0x54a4U,
    0x54a5U, 0x54aaU, 0x5501U, 0x5502U, 0x5504U, 0x5505U, 0x5506U, 0x5509U, 0x5510U, 0x5511U, 0x5512U, 0x5514U,
    0x5515U, 0x5516U, 0x5519U, 0x551aU, 0x5521U, 0x5524U, 0x5525U, 0x5526U, 0x5529U, 0x5540U, 0x5541U, 0x5542U,
    0x5544U, 0x5545U, 0x5546U, 0x5548U, 0x5549U, 0x5550U, 0x5551U, 0x5552U, 0x5554U, 0x5555U, 0x5556U, 0x5558U,
    0x5559U, 0x555aU, 0x5560U, 0x5561U, 0x5564U, 0x5565U, 0x5566U, 0x5568U, 0x5569U, 0x556aU, 0x5581U, 0x5584U,
    0x5585U, 0x5589U, 0x558aU, 0x5590U, 0x5591U, 0x5594U, 0x5595U, 0x5596U, 0x5598U, 0x5599U, 0x55a1U, 0x55a4U,
    0x55a5U, 0x55a6U, 0x55a9U, 0x5600U, 0x5601U, 0x5602U, 0x5604U, 0x5606U, 0x5608U, 0x5609U, 0x5611U, 0x5614U,
    0x5615U, 0x5618U, 0x5619U, 0x5620U, 0x5621U, 0x5622U, 0x5624U, 0x5625U, 0x5626U, 0x5628U, 0x5629U, 0x5641U,
    0x5645U, 0x5646U, 0x5648U, 0x5649U, 0x564aU, 0x5650U, 0x5651U, 0x5652U, 0x5654U, 0x5655U, 0x5656U, 0x5658U,
    0x5659U, 0x565aU, 0x5661U, 0x5664U, 0x5665U, 0x5669U, 0x5682U, 0x5685U, 0x5686U, 0x5688U, 0x5689U, 0x568aU,
    0x5691U, 0x5695U, 0x569aU, 0x56a2U, 0x56a5U, 0x56a6U, 0x56a8U, 0x56a9U, 0x5804U, 0x5805U, 0x5806U, 0x5809U,
    0x5810U, 0x5815U, 0x5818U, 0x5821U, 0x582aU, 0x5845U, 0x5848U, 0x584aU, 0x5851U, 0x5854U, 0x5855U, 0x5856U,
    0x5858U, 0x5859U, 0x5860U, 0x5862U, 0x5864U, 0x5865U, 0x5882U, 0x5889U, 0x5890U, 0x5892U, 0x5895U, 0x5898U,
    0x58a1U, 0x58a9U, 0x5901U, 0x5902U, 0x5905U, 0x590aU, 0x5911U, 0x5914U, 0x5915U, 0x5916U, 0x5919U, 0x5925U,
    0x5941U, 0x5944U, 0x5945U, 0x5946U, 0x5949U, 0x5950U, 0x5951U, 0x5952U, 0x5954U, 0x5955U, 0x5956U, 0x5958U,
    0x5959U, 0x595aU, 0x5961U, 0x5964U, 0x5965U, 0x5966U, 0x5969U, 0x5981U, 0x5985U, 0x5989U, 0x5991U, 0x5994U,
    0x5995U, 0x5996U, 0x5998U, 0x5999U, 0x59a5U, 0x5a04U, 0x5a08U, 0x5a15U, 0x5a1aU, 0x5a20U, 0x5a25U, 0x5a26U,
    0x5a29U, 0x5a45U, 0x5a48U, 0x5a49U, 0x5a51U, 0x5a55U, 0x5a56U, 0x5a58U, 0x5a59U, 0x5a62U, 0x5a65U, 0x5a68U,
    0x5a6aU, 0x5a81U, 0x5a8aU, 0x5a92U, 0x5a95U, 0x5a96U, 0x5a98U, 0x5a9aU, 0x5aa1U, 0x6005U, 0x6014U, 0x6016U,
    0x6019U, 0x6025U, 0x6044U, 0x6050U, 0x6055U, 0x6056U, 0x6058U, 0x605aU, 0x6061U, 0x6064U, 0x6066U, 0x6069U,
    0x6081U, 0x6096U, 0x60a5U, 0x6101U, 0x6104U, 0x6106U, 0x6109U, 0x6112U, 0x6115U, 0x6121U, 0x6122U, 0x6126U,
    0x6129U, 0x6145U, 0x6149U, 0x6151U, 0x6155U, 0x6156U, 0x6159U, 0x6165U, 0x6166U, 0x616aU, 0x6184U, 0x618aU,
    0x6192U, 0x6195U, 0x61a1U, 0x61a6U, 0x61a9U, 0x6211U, 0x6216U, 0x6219U, 0x6240U, 0x6241U, 0x6246U, 0x6255U,
    0x6256U, 0x6258U, 0x6260U, 0x6285U, 0x6291U, 0x6296U, 0x62a5U, 0x6411U, 0x6412U, 0x6415U, 0x6416U, 0x641aU,
    0x6421U, 0x6426U, 0x6429U, 0x6440U, 0x6442U, 0x6445U, 0x6448U, 0x644aU, 0x6451U, 0x6454U, 0x6455U, 0x6456U,
    0x6459U, 0x645aU, 0x6460U, 0x6462U, 0x6465U, 0x6484U, 0x6485U, 0x6489U, 0x6490U, 0x6492U, 0x6494U, 0x6495U,
    0x6496U, 0x6498U, 0x649aU, 0x64a1U, 0x64a4U, 0x64a9U, 0x6505U, 0x6508U, 0x650aU, 0x6511U, 0x6515U, 0x6516U,
    0x6519U, 0x6544U, 0x6545U, 0x6546U, 0x6549U, 0x6550U, 0x6551U, 0x6554U, 0x6555U, 0x6556U, 0x6559U, 0x6561U,
    0x6564U, 0x6565U, 0x6566U, 0x6569U, 0x6586U, 0x6589U, 0x658aU, 0x6591U, 0x6595U, 0x6596U, 0x6599U, 0x659aU,
    0x65a2U, 0x65a5U, 0x65a6U, 0x65a8U, 0x6602U, 0x6609U, 0x6615U, 0x6620U, 0x6626U, 0x6628U, 0x6629U, 0x6640U,
    0x6645U, 0x6648U, 0x664aU, 0x6651U, 0x6654U, 0x6655U, 0x6656U, 0x6658U, 0x665aU, 0x6660U, 0x6665U, 0x6668U,
    0x6680U, 0x6682U, 0x6685U, 0x668aU, 0x6694U, 0x6696U, 0x6698U, 0x6699U, 0x66a0U, 0x66a4U, 0x66a6U, 0x66aaU,
    0x6816U, 0x6819U, 0x6825U, 0x6841U, 0x6852U, 0x6855U, 0x685aU, 0x6861U, 0x6869U, 0x6885U, 0x6891U, 0x6898U,
    0x68a6U, 0x6901U, 0x6904U, 0x6910U, 0x6915U, 0x6921U, 0x6924U, 0x6926U, 0x6929U, 0x6940U, 0x6941U, 0x6945U,
    0x6946U, 0x6948U, 0x6951U, 0x6954U, 0x6955U, 0x6956U, 0x6959U, 0x6960U, 0x6965U, 0x696aU, 0x6982U, 0x6984U,
    0x698aU, 0x6995U, 0x69a1U, 0x69a4U, 0x69a5U, 0x69a9U, 0x6a11U, 0x6a16U, 0x6a18U, 0x6a41U, 0x6a44U, 0x6a49U,
    0x6a50U, 0x6a55U, 0x6a58U, 0x6a5aU, 0x6a64U, 0x6a65U, 0x6a69U, 0x6a86U, 0x6a94U, 0x6a98U, 0x6a9aU, 0x6aa6U,
    0x8000U, 0x8002U, 0x8008U, 0x800aU, 0x8020U, 0x8022U, 0x8028U, 0x802aU, 0x8045U, 0x8050U, 0x8051U, 0x8054U,
    0x8056U, 0x8059U, 0x8065U, 0x8080U, 0x8082U, 0x8088U, 0x808aU, 0x8095U, 0x80a0U, 0x80a2U, 0x80a8U, 0x80aaU,
    0x8105U, 0x8111U, 0x8114U, 0x8116U, 0x8119U, 0x8125U, 0x8141U, 0x8144U, 0x8149U, 0x8150U, 0x8152U, 0x8155U,
    0x8156U, 0x8158U, 0x8159U, 0x8164U, 0x8166U, 0x8169U, 0x8185U, 0x8189U, 0x8194U, 0x8196U, 0x8199U, 0x81a5U,
    0x8200U, 0x8202U, 0x8208U, 0x820aU, 0x8215U, 0x8220U, 0x8222U, 0x8228U, 0x822aU, 0x8251U, 0x8254U, 0x8259U,
    0x8265U, 0x8280U, 0x8282U, 0x8288U, 0x828aU, 0x8295U, 0x82a0U, 0x82a2U, 0x82a8U, 0x82aaU, 0x8414U, 0x8419U,
    0x8441U, 0x8444U, 0x8451U, 0x8455U, 0x845aU, 0x8461U, 0x8464U, 0x8469U, 0x8494U, 0x8499U, 0x8501U, 0x8509U,
    0x8512U, 0x8515U, 0x851aU, 0x8526U, 0x8529U, 0x8540U, 0x8541U, 0x8545U, 0x8548U, 0x8551U, 0x8554U, 0x8555U,
    0x8556U, 0x8559U, 0x855aU, 0x8565U, 0x8566U, 0x8568U, 0x856aU, 0x8581U, 0x8584U, 0x8586U, 0x8589U, 0x8590U,
    0x8592U, 0x8595U, 0x8598U, 0x85a6U, 0x8611U, 0x8616U, 0x8619U, 0x8625U, 0x8641U, 0x8644U, 0x8649U, 0x864aU,
    0x8650U, 0x8655U, 0x8659U, 0x865aU, 0x8661U, 0x8666U, 0x866aU, 0x8685U, 0x8691U, 0x869aU, 0x86a4U, 0x8800U,
    0x8802U, 0x8808U, 0x880aU, 0x8815U, 0x8820U, 0x8822U, 0x8828U, 0x882aU, 0x8841U, 0x8845U, 0x8851U, 0x8854U,
    0x8859U, 0x8865U, 0x8869U, 0x8880U, 0x8882U, 0x8888U, 0x888aU, 0x8895U, 0x88a0U, 0x88a2U, 0x88a8U, 0x88aaU,
    0x8905U, 0x8906U, 0x8911U, 0x8914U, 0x8916U, 0x8925U, 0x8941U, 0x8944U, 0x8946U, 0x8949U, 0x8950U, 0x8952U,
    0x8955U, 0x895aU, 0x8961U, 0x8964U, 0x8985U, 0x8996U, 0x8999U, 0x89a5U, 0x8a00U, 0x8a02U, 0x8a08U, 0x8a0aU,
    0x8a15U, 0x8a20U, 0x8a22U, 0x8a28U, 0x8a2aU, 0x8a45U, 0x8a51U, 0x8a54U, 0x8a56U, 0x8a80U, 0x8a82U, 0x8a88U,
    0x8a8aU, 0x8a95U, 0x8aa0U, 0x8aa2U, 0x8aa8U, 0x8aaaU, 0x9005U, 0x9011U, 0x9016U, 0x9018U, 0x9019U, 0x9025U,
    0x9041U, 0x9046U, 0x9049U, 0x9055U, 0x9058U, 0x905aU, 0x9069U, 0x906aU, 0x9085U, 0x9091U, 0x9094U, 0x9096U,
    0x9099U, 0x90a5U, 0x9101U, 0x9104U, 0x9106U, 0x9109U, 0x9110U, 0x9115U, 0x9118U, 0x911aU, 0x9121U, 0x9124U,
    0x9126U, 0x9129U, 0x9140U, 0x9145U, 0x9150U, 0x9151U, 0x9154U, 0x9155U, 0x9156U, 0x9159U, 0x9162U, 0x9165U,
    0x9184U, 0x9186U, 0x9192U, 0x9195U, 0x9198U, 0x91a1U, 0x91a4U, 0x91a6U, 0x91a9U, 0x9205U, 0x9211U, 0x9214U,
    0x9219U, 0x9225U, 0x9244U, 0x9246U, 0x9249U, 0x9250U, 0x9252U, 0x9255U, 0x9258U, 0x9266U, 0x9269U, 0x9285U,
    0x9294U, 0x9296U, 0x92a9U, 0x9401U, 0x9404U, 0x9406U, 0x9410U, 0x9415U, 0x9418U, 0x9426U, 0x9440U, 0x944aU,
    0x9451U, 0x9454U, 0x9455U, 0x9456U, 0x9458U, 0x9459U, 0x9460U, 0x9461U, 0x9462U, 0x9465U, 0x9484U, 0x9486U,
    0x9492U, 0x9494U, 0x9495U, 0x9498U, 0x94a1U, 0x94a9U, 0x9500U, 0x9505U, 0x9508U, 0x950aU, 0x9510U, 0x9511U,
    0x9514U, 0x9515U, 0x9516U, 0x9519U, 0x9521U, 0x9525U, 0x9529U, 0x952aU, 0x9541U, 0x9544U, 0x9545U, 0x9546U,
    0x9549U, 0x9550U, 0x9551U, 0x9552U, 0x9554U, 0x9555U, 0x9556U, 0x9558U, 0x9559U, 0x955aU, 0x9561U, 0x9564U,
    0x9565U, 0x9566U, 0x9569U, 0x9581U, 0x9585U, 0x9588U, 0x9591U, 0x9592U, 0x9594U, 0x9595U, 0x9596U, 0x9599U,
    0x959aU, 0x95a0U, 0x95a2U, 0x95a5U, 0x95a8U, 0x95aaU, 0x9601U, 0x9604U, 0x9610U, 0x9615U, 0x9619U, 0x9620U,
    0x9626U, 0x9629U, 0x9645U, 0x9648U, 0x9649U, 0x9651U, 0x9652U, 0x9655U, 0x9656U, 0x9659U, 0x9665U, 0x9668U,
    0x9682U, 0x9684U, 0x9689U, 0x968aU, 0x9692U, 0x9694U, 0x9695U, 0x96a4U, 0x96a6U, 0x96a9U, 0x9805U, 0x9816U,
    0x9819U, 0x9825U, 0x9841U, 0x9846U, 0x9850U, 0x9852U, 0x9855U, 0x9856U, 0x985aU, 0x9864U, 0x9865U, 0x9885U,
    0x9891U, 0x9896U, 0x9899U, 0x98a5U, 0x9904U, 0x9906U, 0x9909U, 0x9910U, 0x9912U, 0x9915U, 0x9918U, 0x991aU,
    0x9920U, 0x9921U, 0x9924U, 0x9926U, 0x9940U, 0x9942U, 0x9945U, 0x9948U, 0x994aU, 0x9951U, 0x9954U, 0x9955U,
    0x9956U, 0x9959U, 0x9962U, 0x9965U, 0x9966U, 0x996aU, 0x9981U, 0x9984U, 0x9990U, 0x9992U, 0x9995U, 0x999aU,
    0x99a1U, 0x99a6U, 0x9a05U, 0x9a15U, 0x9a25U, 0x9a44U, 0x9a46U, 0x9a49U, 0x9a50U, 0x9a55U, 0x9a58U, 0x9a61U,
    0x9a85U, 0x9a91U, 0x9a94U, 0x9a95U, 0x9a96U, 0xa000U, 0xa002U, 0xa008U, 0xa00aU, 0xa015U, 0xa020U, 0xa022U,
    0xa028U, 0xa02aU, 0xa045U, 0xa051U, 0xa054U, 0xa056U, 0xa059U, 0xa080U, 0xa082U, 0xa088U, 0xa08aU, 0xa095U,
    0xa0a0U, 0xa0a2U, 0xa0a8U, 0xa0aaU, 0xa105U, 0xa109U, 0xa111U, 0xa114U, 0xa116U, 0xa119U, 0xa11aU, 0xa146U,
    0xa149U, 0xa151U, 0xa155U, 0xa158U, 0xa15aU, 0xa161U, 0xa164U, 0xa185U, 0xa190U, 0xa192U, 0xa196U, 0xa199U,
    0xa202U, 0xa208U, 0xa20aU, 0xa210U, 0xa219U, 0xa222U, 0xa228U, 0xa22aU, 0xa245U, 0xa251U, 0xa256U, 0xa259U,
    0xa265U, 0xa280U, 0xa282U, 0xa288U, 0xa28aU, 0xa295U, 0xa2a0U, 0xa2a2U, 0xa2a8U, 0xa2aaU, 0xa419U, 0xa425U,
    0xa441U, 0xa444U, 0xa450U, 0xa454U, 0xa455U, 0xa458U, 0xa45aU, 0xa461U, 0xa465U, 0xa466U, 0xa468U, 0xa469U,
    0xa485U, 0xa506U, 0xa509U, 0xa510U, 0xa512U, 0xa515U, 0xa518U, 0xa526U, 0xa529U, 0xa542U, 0xa545U, 0xa551U,
    0xa554U, 0xa555U, 0xa556U, 0xa559U, 0xa565U, 0xa56aU, 0xa581U, 0xa584U, 0xa585U, 0xa586U, 0xa589U, 0xa592U,
    0xa595U, 0xa598U, 0xa605U, 0xa611U, 0xa616U, 0xa61aU, 0xa621U, 0xa625U, 0xa644U, 0xa646U, 0xa64aU, 0xa652U,
    0xa655U, 0xa656U, 0xa658U, 0xa660U, 0xa662U, 0xa686U, 0xa690U, 0xa695U, 0xa696U, 0xa699U, 0xa6a1U, 0xa6a4U,
    0xa6a6U, 0xa800U, 0xa802U, 0xa808U, 0xa80aU, 0xa820U, 0xa822U, 0xa828U, 0xa82aU, 0xa851U, 0xa854U, 0xa856U,
    0xa859U, 0xa880U, 0xa882U, 0xa888U, 0xa88aU, 0xa895U, 0xa8a0U, 0xa8a2U, 0xa8a8U, 0xa8aaU, 0xa905U, 0xa914U,
    0xa919U, 0xa921U, 0xa925U, 0xa941U, 0xa950U, 0xa955U, 0xa95aU, 0xa961U, 0xa966U, 0xa969U, 0xa990U, 0xa996U,
    0xaa00U, 0xaa02U, 0xaa08U, 0xaa0aU, 0xaa20U, 0xaa22U, 0xaa28U, 0xaa2aU, 0xaa51U, 0xaa54U, 0xaa56U, 0xaa80U,
    0xaa82U, 0xaa88U, 0xaa8aU, 0xaa95U, 0xaaa0U, 0xaaa2U, 0xaaa8U, 0xaaaaU,
};
static_assert(kIq1Packed[0] == 0x0000U);
static_assert(kIq1Packed[1] == 0x0002U);
static_assert(kIq1Packed[2] == 0x0005U);
static_assert(kIq1Packed[2047] == 0xaaaaU);

std::size_t matrix_bytes(QType qtype, std::int32_t rows, std::int32_t columns) {
    std::size_t values = 0;
    std::size_t bytes = 0;
    switch (qtype) {
    case QType::GGML_IQ1_S:
        values = kIq1BlockValues;
        bytes = kIq1BlockBytes;
        break;
    case QType::GGML_IQ4_NL:
        values = kIq4BlockValues;
        bytes = kIq4BlockBytes;
        break;
    case QType::GGML_Q5_K:
        values = kQ5BlockValues;
        bytes = kQ5BlockBytes;
        break;
    case QType::GGML_Q8_0:
        values = kQ8BlockValues;
        bytes = kQ8BlockBytes;
        break;
    default:
        throw std::logic_error("real sparse-MoE oracle encountered an unsupported GGML format");
    }
    if (columns <= 0 || rows <= 0 || static_cast<std::size_t>(columns) % values != 0) {
        throw std::logic_error("real sparse-MoE oracle encountered malformed matrix geometry");
    }
    return static_cast<std::size_t>(rows) * (static_cast<std::size_t>(columns) / values) * bytes;
}

double iq1_value(const std::uint8_t* block, std::int32_t index) {
    const int group = index / 32;
    const int lane = (index / 8) & 3;
    const int item = index & 7;
    const std::uint16_t control = read_u16(block + 34 + 2 * group);
    const int grid = block[2 + 4 * group + lane] |
                     (((control >> (3 * lane)) & 7U) << 8U);
    const int ternary = (kIq1Packed[static_cast<std::size_t>(grid)] >> (2 * item)) & 3U;
    if (ternary > 2) { throw std::logic_error("invalid IQ1_S oracle codebook digit"); }
    const int digit = ternary - 1;
    const double delta = (control & 0x8000U) != 0 ? -0.125 : 0.125;
    const int multiplier = 2 * ((control >> 12U) & 7U) + 1;
    return binary16_to_double(read_u16(block)) * multiplier * (digit + delta);
}

double iq4_value(const std::uint8_t* block, std::int32_t index) {
    const std::uint8_t packed = block[2 + (index & 15)];
    const int code = index < 16 ? packed & 15U : packed >> 4U;
    return binary16_to_double(read_u16(block)) * kIq4Nl[static_cast<std::size_t>(code)];
}

double decode_value(QType qtype, const std::uint8_t* block, std::int32_t index) {
    switch (qtype) {
    case QType::GGML_IQ1_S:
        return iq1_value(block, index);
    case QType::GGML_IQ4_NL:
        return iq4_value(block, index);
    case QType::GGML_Q5_K:
        return ggml_q5_k_value(block, index);
    case QType::GGML_Q8_0:
        return ggml_q8_0_value(block, index);
    default:
        throw std::logic_error("real sparse-MoE oracle encountered an unsupported decoder");
    }
}

std::pair<std::size_t, std::size_t> block_shape(QType qtype) {
    switch (qtype) {
    case QType::GGML_IQ1_S:
        return {kIq1BlockValues, kIq1BlockBytes};
    case QType::GGML_IQ4_NL:
        return {kIq4BlockValues, kIq4BlockBytes};
    case QType::GGML_Q5_K:
        return {kQ5BlockValues, kQ5BlockBytes};
    case QType::GGML_Q8_0:
        return {kQ8BlockValues, kQ8BlockBytes};
    default:
        throw std::logic_error("real sparse-MoE oracle encountered an unsupported block shape");
    }
}

std::vector<double> linear(QType qtype, std::span<const std::uint8_t> matrix,
                           std::int32_t rows, std::int32_t columns,
                           std::span<const double> input) {
    const auto [block_values, block_bytes] = block_shape(qtype);
    const std::size_t row_bytes = matrix_bytes(qtype, 1, columns);
    if (matrix.size() != static_cast<std::size_t>(rows) * row_bytes ||
        input.size() != static_cast<std::size_t>(columns)) {
        throw std::logic_error("real sparse-MoE oracle received a malformed matrix");
    }
    std::vector<double> result(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto* row_data = matrix.data() + static_cast<std::size_t>(row) * row_bytes;
        double sum = 0.0;
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto* block = row_data +
                (static_cast<std::size_t>(column) / block_values) * block_bytes;
            sum += decode_value(qtype, block,
                                column % static_cast<std::int32_t>(block_values)) *
                   input[static_cast<std::size_t>(column)];
        }
        result[static_cast<std::size_t>(row)] = sum;
    }
    return result;
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double silu(double value) { return value * sigmoid(value); }

struct GrOracle {
    std::vector<double> mixed;
    std::vector<double> write_scale;
};

GrOracle gr_oracle(std::span<const std::uint16_t> residual_bits,
                   std::span<const float> gamma, std::span<const std::uint8_t> down,
                   std::span<const std::uint8_t> up, std::span<const float> write) {
    if (residual_bits.size() != static_cast<std::size_t>(kFlat) ||
        gamma.size() != static_cast<std::size_t>(kFlat) ||
        write.size() != static_cast<std::size_t>(kBranches) * kFlat) {
        throw std::logic_error("real accumulated GR oracle received malformed represented data");
    }
    std::vector<double> normalized(static_cast<std::size_t>(kFlat));
    for (std::int32_t branch = 0; branch < kBranches; ++branch) {
        const std::size_t base = static_cast<std::size_t>(branch) * kHidden;
        double sum_squares = 0.0;
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            const double value = bf16_to_f32(residual_bits[base + dimension]);
            sum_squares += value * value;
        }
        const double inverse_rms = 1.0 / std::sqrt(sum_squares / kHidden + 1.0e-6);
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            const std::size_t index = base + dimension;
            // The actual GGUF has already folded the source checkpoint's unit offset. Gamma is
            // therefore consumed directly here, matching the public native Op contract.
            normalized[index] = static_cast<double>(bf16_to_f32(residual_bits[index])) *
                                inverse_rms * static_cast<double>(gamma[index]);
        }
    }

    std::vector<double> low_rank =
        linear(QType::GGML_Q8_0, down, kGrRank, kFlat, normalized);
    for (double& value : low_rank) { value = silu(value / kBranches); }
    const std::vector<double> gate_logits =
        linear(QType::GGML_Q8_0, up, kFlat, kGrRank, low_rank);

    GrOracle result{std::vector<double>(static_cast<std::size_t>(kHidden)),
                    std::vector<double>(static_cast<std::size_t>(kBranches))};
    for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
        double mixed = 0.0;
        for (std::int32_t branch = 0; branch < kBranches; ++branch) {
            const std::size_t index =
                static_cast<std::size_t>(branch) * kHidden + dimension;
            mixed += sigmoid(gate_logits[index]) * normalized[index];
        }
        result.mixed[static_cast<std::size_t>(dimension)] = mixed / kBranches;
    }
    for (std::int32_t branch = 0; branch < kBranches; ++branch) {
        double projected = 0.0;
        const std::size_t row = static_cast<std::size_t>(branch) * kFlat;
        for (std::int32_t column = 0; column < kFlat; ++column) {
            projected += static_cast<double>(write[row + column]) *
                         normalized[static_cast<std::size_t>(column)];
        }
        result.write_scale[static_cast<std::size_t>(branch)] =
            2.0 * sigmoid(projected / kBranches);
    }
    return result;
}

struct OracleRoute {
    std::array<std::int32_t, kTopK> ids{};
    std::array<double, kTopK> weights{};
};

OracleRoute route_oracle(std::span<const float> router, std::span<const double> input) {
    if (router.size() != static_cast<std::size_t>(kExperts) * kHidden ||
        input.size() != static_cast<std::size_t>(kHidden)) {
        throw std::logic_error("real sparse-MoE route oracle received malformed represented data");
    }
    std::array<double, kExperts> logits{};
    for (std::int32_t expert = 0; expert < kExperts; ++expert) {
        double logit = 0.0;
        for (std::int32_t column = 0; column < kHidden; ++column) {
            logit += static_cast<double>(
                         router[static_cast<std::size_t>(expert) * kHidden + column]) *
                     input[static_cast<std::size_t>(column)];
        }
        if (!std::isfinite(logit)) {
            throw std::logic_error("real sparse-MoE route oracle produced a non-finite logit");
        }
        logits[static_cast<std::size_t>(expert)] = logit;
    }
    std::array<std::int32_t, kExperts> order{};
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::int32_t left, std::int32_t right) {
        const double lhs = logits[static_cast<std::size_t>(left)];
        const double rhs = logits[static_cast<std::size_t>(right)];
        return lhs != rhs ? lhs > rhs : left < right;
    });

    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::array<double, kExperts> probabilities{};
    double denominator = 0.0;
    for (std::int32_t expert = 0; expert < kExperts; ++expert) {
        const double probability =
            std::exp(logits[static_cast<std::size_t>(expert)] - maximum);
        probabilities[static_cast<std::size_t>(expert)] = probability;
        denominator += probability;
    }
    OracleRoute route;
    double selected_sum = 0.0;
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        route.ids[static_cast<std::size_t>(rank)] = order[static_cast<std::size_t>(rank)];
        route.weights[static_cast<std::size_t>(rank)] =
            probabilities[static_cast<std::size_t>(route.ids[static_cast<std::size_t>(rank)])] /
            denominator;
        selected_sum += route.weights[static_cast<std::size_t>(rank)];
    }
    for (double& weight : route.weights) { weight /= selected_sum; }
    return route;
}

std::span<const std::uint8_t> byte_span(std::span<const std::byte> bytes) {
    return {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()};
}

std::span<const std::uint8_t> expert_span(std::span<const std::byte> bank,
                                          std::int32_t expert, std::size_t matrix_size) {
    if (expert < 0 || expert >= kExperts ||
        bank.size() != static_cast<std::size_t>(kExperts) * matrix_size) {
        throw std::logic_error("real sparse-MoE oracle received a malformed routed bank");
    }
    return byte_span(bank.subspan(static_cast<std::size_t>(expert) * matrix_size, matrix_size));
}

std::vector<std::uint8_t> selected_device_expert(const Weight& bank, std::int32_t expert,
                                                 std::size_t matrix_size) {
    if (expert < 0 || expert >= kExperts || bank.qdata == nullptr ||
        bank.payload_bytes != static_cast<std::uint64_t>(kExperts) * matrix_size) {
        throw std::logic_error("real sparse-MoE oracle received a malformed device expert bank");
    }
    return copy_device_bytes(static_cast<const std::uint8_t*>(bank.qdata) +
                                 static_cast<std::size_t>(expert) * matrix_size,
                             matrix_size);
}

struct OracleWeights {
    std::span<const float> router;
    const ops::Qwen4MappedRoutedGateUp& routed_gate_up;
    const std::array<std::vector<std::uint8_t>, kTopK>& routed_down;
    std::span<const float> shared_gate;
    std::span<const std::uint8_t> shared_gate_proj;
    std::span<const std::uint8_t> shared_up;
    std::span<const std::uint8_t> shared_down;
};

std::vector<double> complete_oracle(std::span<const double> input, const OracleRoute& route,
                                    const OracleWeights& weights) {
    const std::size_t routed_matrix =
        matrix_bytes(QType::GGML_IQ1_S, kIntermediate, kHidden);
    std::vector<double> output(static_cast<std::size_t>(kHidden), 0.0);
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        const std::int32_t expert = route.ids[static_cast<std::size_t>(rank)];
        const std::vector<double> gate =
            linear(QType::GGML_IQ1_S,
                   expert_span(weights.routed_gate_up.gate, expert, routed_matrix),
                   kIntermediate, kHidden, input);
        const std::vector<double> up =
            linear(QType::GGML_IQ1_S,
                   expert_span(weights.routed_gate_up.up, expert, routed_matrix),
                   kIntermediate, kHidden, input);
        std::vector<double> activated(static_cast<std::size_t>(kIntermediate));
        for (std::int32_t row = 0; row < kIntermediate; ++row) {
            activated[static_cast<std::size_t>(row)] =
                silu(gate[static_cast<std::size_t>(row)]) * up[static_cast<std::size_t>(row)];
        }
        const std::vector<double> down =
            linear(QType::GGML_IQ4_NL, weights.routed_down[static_cast<std::size_t>(rank)],
                   kHidden, kIntermediate, activated);
        for (std::int32_t row = 0; row < kHidden; ++row) {
            output[static_cast<std::size_t>(row)] +=
                route.weights[static_cast<std::size_t>(rank)] *
                down[static_cast<std::size_t>(row)];
        }
    }

    const std::vector<double> shared_gate_projection =
        linear(QType::GGML_Q5_K, weights.shared_gate_proj, kIntermediate, kHidden, input);
    const std::vector<double> shared_up =
        linear(QType::GGML_Q5_K, weights.shared_up, kIntermediate, kHidden, input);
    std::vector<double> shared_activated(static_cast<std::size_t>(kIntermediate));
    for (std::int32_t row = 0; row < kIntermediate; ++row) {
        shared_activated[static_cast<std::size_t>(row)] =
            silu(shared_gate_projection[static_cast<std::size_t>(row)]) *
            shared_up[static_cast<std::size_t>(row)];
    }
    const std::vector<double> shared =
        linear(QType::GGML_Q8_0, weights.shared_down, kHidden, kIntermediate,
               shared_activated);
    double gate_logit = 0.0;
    for (std::int32_t column = 0; column < kHidden; ++column) {
        gate_logit += static_cast<double>(weights.shared_gate[static_cast<std::size_t>(column)]) *
                      input[static_cast<std::size_t>(column)];
    }
    const double shared_weight = sigmoid(gate_logit);
    for (std::int32_t row = 0; row < kHidden; ++row) {
        output[static_cast<std::size_t>(row)] +=
            shared_weight * shared[static_cast<std::size_t>(row)];
    }
    return output;
}

void require_ggml_weight(const Weight& weight, QType qtype, std::uint32_t ndim,
                         std::array<std::int32_t, 3> shape, std::size_t expected_bytes,
                         std::string_view label) {
    const auto [block_values, block_bytes] = block_shape(qtype);
    (void)block_bytes;
    const std::int32_t expected_n = ndim == 2 ? shape[0] : shape[1];
    const std::int32_t expected_k = ndim == 2 ? shape[1] : shape[2];
    if (weight.qtype != qtype || weight.qdata == nullptr || weight.payload != weight.qdata ||
        weight.payload_bytes != expected_bytes || weight.qhigh != nullptr ||
        weight.scales != nullptr || weight.high_plane_bytes != 0 || weight.ndim != ndim ||
        weight.layout != ninfer::QuantLayout::GgmlBlockRow || weight.shape[0] != shape[0] ||
        weight.shape[1] != shape[1] || weight.shape[2] != shape[2] ||
        weight.padded_shape[0] != shape[0] || weight.padded_shape[1] != shape[1] ||
        weight.padded_shape[2] != shape[2] || weight.n != expected_n ||
        weight.k != expected_k || weight.group_size != block_values ||
        weight.group != static_cast<std::int32_t>(block_values)) {
        throw std::logic_error("Qwen4 real oracle binding changed for " + std::string(label));
    }
}

void validate_moe_weights(const ops::Qwen4SparseMoeWeights& weights,
                          std::string_view label) {
    const std::size_t routed_matrix =
        matrix_bytes(QType::GGML_IQ1_S, kIntermediate, kHidden);
    const std::size_t routed_down_matrix =
        matrix_bytes(QType::GGML_IQ4_NL, kHidden, kIntermediate);
    if (weights.router.qtype != QType::FP32_CTRL ||
        weights.router.layout != ninfer::QuantLayout::Contiguous ||
        weights.router.payload == nullptr || weights.router.qdata != weights.router.payload ||
        weights.router.payload_bytes != static_cast<std::uint64_t>(kExperts) * kHidden *
                                               sizeof(float) ||
        weights.router.qhigh != nullptr || weights.router.scales != nullptr ||
        weights.router.high_plane_bytes != 0 || weights.router.n != kExperts ||
        weights.router.k != kHidden ||
        weights.router.ndim != 2 || weights.router.shape[0] != kExperts ||
        weights.router.shape[1] != kHidden || weights.router.padded_shape[0] != kExperts ||
        weights.router.padded_shape[1] != kHidden ||
        weights.routed_gate_up.qtype != QType::GGML_IQ1_S ||
        weights.routed_gate_up.gate.size() != static_cast<std::size_t>(kExperts) * routed_matrix ||
        weights.routed_gate_up.up.size() != static_cast<std::size_t>(kExperts) * routed_matrix ||
        weights.shared_gate.dtype != DType::FP32 || weights.shared_gate.numel() != kHidden) {
        throw std::logic_error("Qwen4 " + std::string(label) + " sparse-MoE binding changed");
    }
    require_ggml_weight(weights.routed_down, QType::GGML_IQ4_NL, 3,
                        {kExperts, kHidden, kIntermediate},
                        static_cast<std::size_t>(kExperts) * routed_down_matrix,
                        std::string(label) + " routed_down");
    require_ggml_weight(weights.shared_gate_proj, QType::GGML_Q5_K, 2,
                        {kIntermediate, kHidden, 1},
                        matrix_bytes(QType::GGML_Q5_K, kIntermediate, kHidden),
                        std::string(label) + " shared_gate_proj");
    require_ggml_weight(weights.shared_up, QType::GGML_Q5_K, 2,
                        {kIntermediate, kHidden, 1},
                        matrix_bytes(QType::GGML_Q5_K, kIntermediate, kHidden),
                        std::string(label) + " shared_up");
    require_ggml_weight(weights.shared_down, QType::GGML_Q8_0, 2,
                        {kHidden, kIntermediate, 1},
                        matrix_bytes(QType::GGML_Q8_0, kHidden, kIntermediate),
                        std::string(label) + " shared_down");
}

class PipelineEvents {
public:
    explicit PipelineEvents(cudaStream_t compute_stream) : compute_stream_(compute_stream) {
        try {
            CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&route_ready, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(
                &ids_ready, cudaEventDisableTiming | cudaEventBlockingSync));
            for (std::size_t slot = 0; slot < transfer_ready.size(); ++slot) {
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &transfer_ready[slot], cudaEventDisableTiming | cudaEventBlockingSync));
                CUDA_CHECK(cudaEventCreateWithFlags(&consumer_complete[slot],
                                                     cudaEventDisableTiming));
            }
        } catch (...) {
            destroy();
            throw;
        }
    }

    ~PipelineEvents() { destroy(); }

    PipelineEvents(const PipelineEvents&) = delete;
    PipelineEvents& operator=(const PipelineEvents&) = delete;

    cudaStream_t transfer_stream = nullptr;
    cudaEvent_t route_ready = nullptr;
    cudaEvent_t ids_ready = nullptr;
    std::array<cudaEvent_t, ops::kQwen4SparseMoePipelineSlots> transfer_ready{};
    std::array<cudaEvent_t, ops::kQwen4SparseMoePipelineSlots> consumer_complete{};

private:
    void destroy() noexcept {
        if (compute_stream_ != nullptr) { (void)cudaStreamSynchronize(compute_stream_); }
        if (transfer_stream != nullptr) { (void)cudaStreamSynchronize(transfer_stream); }
        for (cudaEvent_t& event : consumer_complete) {
            if (event != nullptr) { (void)cudaEventDestroy(std::exchange(event, nullptr)); }
        }
        for (cudaEvent_t& event : transfer_ready) {
            if (event != nullptr) { (void)cudaEventDestroy(std::exchange(event, nullptr)); }
        }
        if (ids_ready != nullptr) {
            (void)cudaEventDestroy(std::exchange(ids_ready, nullptr));
        }
        if (route_ready != nullptr) {
            (void)cudaEventDestroy(std::exchange(route_ready, nullptr));
        }
        if (transfer_stream != nullptr) {
            (void)cudaStreamDestroy(std::exchange(transfer_stream, nullptr));
        }
    }

    cudaStream_t compute_stream_ = nullptr;
};

const verifier::GrDiagnosticView& layer_gr(const verifier::TokenResultView& result,
                                           std::size_t layer) {
    const auto found = std::find_if(result.gr.begin(), result.gr.end(),
                                    [layer](const verifier::GrDiagnosticView& item) {
                                        return item.layer == layer;
                                    });
    if (found == result.gr.end()) {
        throw std::logic_error("Qwen4 Program did not expose the requested GR snapshot");
    }
    return *found;
}

const verifier::RouterDiagnosticView& layer_router(const verifier::TokenResultView& result,
                                                   std::size_t layer) {
    const auto found = std::find_if(result.routers.begin(), result.routers.end(),
                                    [layer](const verifier::RouterDiagnosticView& item) {
                                        return item.layer == layer;
                                    });
    if (found == result.routers.end()) {
        throw std::logic_error("Qwen4 Program did not expose the requested router result");
    }
    return *found;
}

void validate_ffn_gr(const verifier::GrWeights& weights, std::string_view label) {
    if (weights.norm.data == nullptr || weights.norm.dtype != DType::FP32 ||
        weights.norm.numel() != kFlat || weights.inject.data == nullptr ||
        weights.inject.dtype != DType::FP32 ||
        weights.inject.numel() != static_cast<std::int64_t>(kBranches) * kFlat) {
        throw std::logic_error("Qwen4 " + std::string(label) + " FFN GR binding changed");
    }
    require_ggml_weight(weights.down, QType::GGML_Q8_0, 2, {kGrRank, kFlat, 1},
                        kQ8DownBytes, std::string(label) + " FFN GR down");
    require_ggml_weight(weights.up, QType::GGML_Q8_0, 2, {kFlat, kGrRank, 1},
                        kQ8UpBytes, std::string(label) + " FFN GR up");
}

struct AccumulatedCapture {
    std::vector<std::uint16_t> attention_residual;
    std::vector<std::uint16_t> ffn_residual;
    std::vector<std::int32_t> router_ids;
    std::vector<float> router_weights;
};

AccumulatedCapture capture_layer35_position221(const verifier::LoadedModel& model,
                                               ninfer::DeviceContext& device) {
    verifier::Program program(model, device, verifier::DiagnosticSnapshots::Enabled);
    program.reset();
    for (std::int32_t position = 0; position < kAccumulatedPosition; ++position) {
        const std::size_t index = static_cast<std::size_t>(position) % kFrozenParagraph.size();
        (void) program.execute_token(kFrozenParagraph[index],
                                     kFrozenParagraph[(index + 1) % kFrozenParagraph.size()]);
    }
    const std::size_t index =
        static_cast<std::size_t>(kAccumulatedPosition) % kFrozenParagraph.size();
    const verifier::TokenResultView result =
        program.execute_token(kFrozenParagraph[index],
                              kFrozenParagraph[(index + 1) % kFrozenParagraph.size()]);
    const verifier::GrDiagnosticView& gr = layer_gr(result, 35);
    const verifier::RouterDiagnosticView& router = layer_router(result, 35);
    if (result.token_index != kAccumulatedPosition ||
        program.frontier() != kAccumulatedPosition + 1 ||
        gr.attention_residual.dtype != DType::BF16 ||
        gr.attention_residual.numel() != kFlat || gr.ffn_residual.dtype != DType::BF16 ||
        gr.ffn_residual.numel() != kFlat || router.selected_ids.dtype != DType::I32 ||
        router.selected_ids.numel() != kTopK ||
        router.selected_weights.dtype != DType::FP32 ||
        router.selected_weights.numel() != kTopK) {
        throw std::logic_error("Qwen4 position-221 layer-35 Program diagnostic changed");
    }
    return {
        .attention_residual = copy_device_values<std::uint16_t>(
            gr.attention_residual.data, static_cast<std::size_t>(kFlat)),
        .ffn_residual = copy_device_values<std::uint16_t>(
            gr.ffn_residual.data, static_cast<std::size_t>(kFlat)),
        .router_ids = copy_device_values<std::int32_t>(router.selected_ids.data, kTopK),
        .router_weights = copy_device_values<float>(router.selected_weights.data, kTopK),
    };
}

int run_accumulated_layer35_cell(const verifier::LoadedModel& model,
                                 ninfer::DeviceContext& device) {
    constexpr std::size_t kLayer = 35;
    const verifier::LayerWeights& layer = model.view().layers[kLayer];
    validate_ffn_gr(layer.ffn_gr, "position-221 layer-35");
    validate_moe_weights(layer.moe, "position-221 layer-35");
    const AccumulatedCapture captured = capture_layer35_position221(model, device);

    const std::vector<float> gamma = copy_device_values<float>(
        layer.ffn_gr.norm.data, static_cast<std::size_t>(kFlat));
    const std::vector<float> write = copy_device_values<float>(
        layer.ffn_gr.inject.data, static_cast<std::size_t>(kBranches) * kFlat);
    const std::vector<std::uint8_t> gr_down =
        copy_device_bytes(layer.ffn_gr.down.qdata, layer.ffn_gr.down.payload_bytes);
    const std::vector<std::uint8_t> gr_up =
        copy_device_bytes(layer.ffn_gr.up.qdata, layer.ffn_gr.up.payload_bytes);
    const GrOracle gr_reference =
        gr_oracle(captured.attention_residual, gamma, gr_down, gr_up, write);

    DeviceBuffer device_residual = to_device(captured.attention_residual);
    GuardedDeviceBuffer device_x(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_scale(static_cast<std::size_t>(kBranches) *
                                     sizeof(std::uint16_t));
    device_x.fill(0xcd);
    device_scale.fill(0xcd);
    Tensor residual(device_residual.p, DType::BF16, {kHidden, kBranches});
    Tensor x(device_x.data(), DType::BF16, {kHidden});
    Tensor scale(device_scale.data(), DType::BF16, {kBranches});
    WorkspaceArena gr_workspace(ops::gated_residual_workspace_capacity_bytes());
    ops::gated_residual_read_write(residual, layer.ffn_gr.norm, layer.ffn_gr.down,
                                   layer.ffn_gr.up, layer.ffn_gr.inject, x, scale,
                                   gr_workspace, device.stream);
    device.synchronize();

    int failures = verify_reduction(
        "Qwen4 real position-221 layer-35 FFN GR mixed",
        from_device_bf16(device_x.data(), kHidden), gr_reference.mixed, kGrReadCriterion);
    failures += verify_reduction(
        "Qwen4 real position-221 layer-35 FFN GR write scale",
        from_device_bf16(device_scale.data(), kBranches), gr_reference.write_scale,
        kGrScaleCriterion);

    const std::vector<std::uint16_t> input_bits =
        from_device<std::uint16_t>(device_x.data(), kHidden);
    std::vector<double> oracle_input(static_cast<std::size_t>(kHidden));
    for (std::size_t dimension = 0; dimension < oracle_input.size(); ++dimension) {
        oracle_input[dimension] = bf16_to_f32(input_bits[dimension]);
        if (!std::isfinite(oracle_input[dimension])) {
            throw std::logic_error("Qwen4 position-221 layer-35 MoE input is non-finite");
        }
    }

    const ops::Qwen4SparseMoeWeights& weights = layer.moe;
    const std::vector<float> router = copy_device_values<float>(
        weights.router.qdata, static_cast<std::size_t>(kExperts) * kHidden);
    const OracleRoute route = route_oracle(router, oracle_input);
    const std::vector<std::int32_t> expected_ids(route.ids.begin(), route.ids.end());
    if (captured.router_ids != expected_ids) {
        std::cerr << "Qwen4 Program position-221 layer-35 selected wrong expert ids\n";
        ++failures;
    }
    failures += verify_pointwise(
        "Qwen4 Program position-221 layer-35 selected weights",
        std::vector<double>(captured.router_weights.begin(), captured.router_weights.end()),
        route.weights, kRouteWeightCriterion);
    const std::size_t routed_down_matrix =
        matrix_bytes(QType::GGML_IQ4_NL, kHidden, kIntermediate);
    std::array<std::vector<std::uint8_t>, kTopK> routed_down;
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        routed_down[static_cast<std::size_t>(rank)] = selected_device_expert(
            weights.routed_down, route.ids[static_cast<std::size_t>(rank)], routed_down_matrix);
    }
    const std::vector<float> shared_gate =
        copy_device_values<float>(weights.shared_gate.data, kHidden);
    const std::vector<std::uint8_t> shared_gate_proj =
        copy_device_bytes(weights.shared_gate_proj.qdata, weights.shared_gate_proj.payload_bytes);
    const std::vector<std::uint8_t> shared_up =
        copy_device_bytes(weights.shared_up.qdata, weights.shared_up.payload_bytes);
    const std::vector<std::uint8_t> shared_down =
        copy_device_bytes(weights.shared_down.qdata, weights.shared_down.payload_bytes);
    const OracleWeights oracle_weights{
        .router = router,
        .routed_gate_up = weights.routed_gate_up,
        .routed_down = routed_down,
        .shared_gate = shared_gate,
        .shared_gate_proj = shared_gate_proj,
        .shared_up = shared_up,
        .shared_down = shared_down,
    };
    const std::vector<double> moe_reference =
        complete_oracle(oracle_input, route, oracle_weights);
    if (!std::all_of(moe_reference.begin(), moe_reference.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::logic_error("Qwen4 position-221 layer-35 MoE oracle is non-finite");
    }

    PinnedHostBuffer pinned(ops::kQwen4SparseMoePipelineStageBytes);
    GuardedDeviceBuffer device_stage(ops::kQwen4SparseMoePipelineStageBytes);
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(kTopK) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_weights(static_cast<std::size_t>(kTopK) * sizeof(float));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(kHidden) *
                                      sizeof(std::uint16_t));
    device_stage.fill(0xcd);
    device_ids.fill(0xcd);
    device_weights.fill(0xcd);
    device_output.fill(0xcd);
    Tensor stage(device_stage.data(), DType::U8,
                 {static_cast<std::int32_t>(ops::kQwen4SparseMoePipelineStageBytes)});
    Tensor selected_ids(device_ids.data(), DType::I32, {kTopK});
    Tensor selected_weights(device_weights.data(), DType::FP32, {kTopK});
    Tensor output(device_output.data(), DType::BF16, {kHidden});
    WorkspaceArena moe_workspace(ops::qwen4_sparse_moe_workspace_capacity_bytes());
    PipelineEvents events(device.stream);
    ops::Qwen4SparseMoePipeline pipeline{
        .pinned_stage = pinned.data(),
        .pinned_stage_bytes = pinned.size(),
        .device_stage = stage,
        .transfer_stream = events.transfer_stream,
        .compute_stream = device.stream,
        .route_ready = events.route_ready,
        .ids_ready = events.ids_ready,
        .transfer_ready = {events.transfer_ready[0], events.transfer_ready[1]},
        .consumer_complete = {events.consumer_complete[0], events.consumer_complete[1]},
    };
    ops::qwen4_sparse_moe(x, weights, pipeline, selected_ids, selected_weights, output,
                          moe_workspace, device.stream);
    device.synchronize();
    CUDA_CHECK(cudaStreamSynchronize(events.transfer_stream));

    const std::vector<std::int32_t> actual_ids =
        from_device<std::int32_t>(device_ids.data(), kTopK);
    if (actual_ids != expected_ids) {
        std::cerr << "Qwen4 real position-221 layer-35 sparse-MoE selected wrong expert ids\n";
        ++failures;
    }
    const std::vector<float> actual_weight_f32 =
        from_device<float>(device_weights.data(), kTopK);
    failures += verify_pointwise(
        "Qwen4 real position-221 layer-35 sparse-MoE selected weights",
        std::vector<double>(actual_weight_f32.begin(), actual_weight_f32.end()), route.weights,
        kRouteWeightCriterion);
    const std::vector<double> actual_moe =
        from_device_bf16(device_output.data(), kHidden);
    failures += verify_reduction(
        "Qwen4 real position-221 layer-35 sparse-MoE complete FP64 formula", actual_moe,
        moe_reference, kOutputCriterion);

    const std::vector<double> represented_scale =
        from_device_bf16(device_scale.data(), kBranches);
    std::vector<double> injected_reference(static_cast<std::size_t>(kFlat));
    for (std::int32_t branch = 0; branch < kBranches; ++branch) {
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            const std::size_t residual_index =
                static_cast<std::size_t>(branch) * kHidden + dimension;
            injected_reference[residual_index] =
                static_cast<double>(bf16_to_f32(captured.attention_residual[residual_index])) +
                represented_scale[static_cast<std::size_t>(branch)] *
                    actual_moe[static_cast<std::size_t>(dimension)];
        }
    }
    std::vector<double> program_ffn(static_cast<std::size_t>(kFlat));
    std::transform(captured.ffn_residual.begin(), captured.ffn_residual.end(),
                   program_ffn.begin(),
                   [](std::uint16_t bits) { return bf16_to_f32(bits); });
    failures += verify_reduction("Qwen4 real position-221 layer-35 post-FFN GR inject",
                                 program_ffn, injected_reference, kGrInjectCriterion);
    failures += device_x.verify_guards("Qwen4 real position-221 layer-35 FFN GR mixed");
    failures += device_scale.verify_guards(
        "Qwen4 real position-221 layer-35 FFN GR write scale");
    failures += device_stage.verify_guards(
        "Qwen4 real position-221 layer-35 sparse-MoE stage");
    failures += device_ids.verify_guards("Qwen4 real position-221 layer-35 sparse-MoE ids");
    failures += device_weights.verify_guards(
        "Qwen4 real position-221 layer-35 sparse-MoE weights");
    failures += device_output.verify_guards(
        "Qwen4 real position-221 layer-35 sparse-MoE output");
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " qwen4_real_position_221_layer_35_ffn_moe_cell\n";
    return failures;
}

} // namespace

namespace ninfer::test::qwen4::real_oracle {

int run_moe_cell(const verifier::LoadedModel& model, ninfer::DeviceContext& device) {
    const ops::Qwen4SparseMoeWeights& weights = model.view().layers[0].moe;
    validate_moe_weights(weights, "layer-0");

    verifier::Program program(model, device, verifier::DiagnosticSnapshots::Enabled);
    program.reset();
    const verifier::TokenResultView result = program.execute_token(48, 16451);
    const verifier::GrDiagnosticView& gr = layer_gr(result, 0);
    if (gr.attention_residual.dtype != DType::BF16 ||
        gr.attention_residual.numel() != static_cast<std::int64_t>(4) * kHidden) {
        throw std::logic_error("Qwen4 layer-0 attention GR snapshot changed");
    }

    GuardedDeviceBuffer device_x(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    device_x.fill(0xcd);
    Tensor x(device_x.data(), DType::BF16, {kHidden});
    WorkspaceArena gr_workspace(ops::gated_residual_workspace_capacity_bytes());
    const verifier::GrWeights& ffn_gr = model.view().layers[0].ffn_gr;
    ops::gated_residual_read(gr.attention_residual, ffn_gr.norm, ffn_gr.down, ffn_gr.up, x,
                             gr_workspace, device.stream);
    device.synchronize();

    const std::vector<std::uint16_t> input_bits =
        from_device<std::uint16_t>(device_x.data(), kHidden);
    std::vector<double> oracle_input(static_cast<std::size_t>(kHidden));
    for (std::size_t index = 0; index < oracle_input.size(); ++index) {
        oracle_input[index] = bf16_to_f32(input_bits[index]);
        if (!std::isfinite(oracle_input[index])) {
            throw std::logic_error("Qwen4 represented layer-0 FFN GR x is non-finite");
        }
    }

    const std::vector<float> router = copy_device_values<float>(
        weights.router.qdata, static_cast<std::size_t>(kExperts) * kHidden);
    const OracleRoute route = route_oracle(router, oracle_input);
    const std::size_t routed_down_matrix =
        matrix_bytes(QType::GGML_IQ4_NL, kHidden, kIntermediate);
    std::array<std::vector<std::uint8_t>, kTopK> routed_down;
    for (std::int32_t rank = 0; rank < kTopK; ++rank) {
        routed_down[static_cast<std::size_t>(rank)] = selected_device_expert(
            weights.routed_down, route.ids[static_cast<std::size_t>(rank)], routed_down_matrix);
    }
    const std::vector<float> shared_gate =
        copy_device_values<float>(weights.shared_gate.data, kHidden);
    const std::vector<std::uint8_t> shared_gate_proj =
        copy_device_bytes(weights.shared_gate_proj.qdata, weights.shared_gate_proj.payload_bytes);
    const std::vector<std::uint8_t> shared_up =
        copy_device_bytes(weights.shared_up.qdata, weights.shared_up.payload_bytes);
    const std::vector<std::uint8_t> shared_down =
        copy_device_bytes(weights.shared_down.qdata, weights.shared_down.payload_bytes);
    const OracleWeights oracle_weights{
        .router = router,
        .routed_gate_up = weights.routed_gate_up,
        .routed_down = routed_down,
        .shared_gate = shared_gate,
        .shared_gate_proj = shared_gate_proj,
        .shared_up = shared_up,
        .shared_down = shared_down,
    };
    // The complete reference is fixed before the production MoE is invoked. Its only production
    // value is the represented public BF16 x above; no route or MoE result feeds this calculation.
    const std::vector<double> reference = complete_oracle(oracle_input, route, oracle_weights);
    if (!std::all_of(reference.begin(), reference.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::logic_error("Qwen4 real layer-0 sparse-MoE oracle output is non-finite");
    }

    constexpr std::size_t kHostGuardBytes = 256;
    PinnedHostBuffer pinned(ops::kQwen4SparseMoePipelineStageBytes + 2 * kHostGuardBytes);
    auto* pinned_bytes = static_cast<std::uint8_t*>(pinned.data());
    std::memset(pinned_bytes, 0xa5, pinned.size());
    void* pinned_stage = pinned_bytes + kHostGuardBytes;
    GuardedDeviceBuffer device_stage(ops::kQwen4SparseMoePipelineStageBytes);
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(kTopK) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_weights(static_cast<std::size_t>(kTopK) * sizeof(float));
    GuardedDeviceBuffer device_destination(static_cast<std::size_t>(kHidden) *
                                           sizeof(std::uint16_t));
    device_stage.fill(0xcd);
    device_ids.fill(0xcd);
    device_weights.fill(0xcd);
    device_destination.fill(0xcd);
    WorkspaceArena moe_workspace(ops::qwen4_sparse_moe_workspace_capacity_bytes());
    PipelineEvents events(device.stream);

    Tensor stage(device_stage.data(), DType::U8,
                 {static_cast<std::int32_t>(ops::kQwen4SparseMoePipelineStageBytes)});
    Tensor selected_ids(device_ids.data(), DType::I32, {kTopK});
    Tensor selected_weights(device_weights.data(), DType::FP32, {kTopK});
    Tensor destination(device_destination.data(), DType::BF16, {kHidden});
    ops::Qwen4SparseMoePipeline pipeline{
        .pinned_stage = pinned_stage,
        .pinned_stage_bytes = ops::kQwen4SparseMoePipelineStageBytes,
        .device_stage = stage,
        .transfer_stream = events.transfer_stream,
        .compute_stream = device.stream,
        .route_ready = events.route_ready,
        .ids_ready = events.ids_ready,
        .transfer_ready = {events.transfer_ready[0], events.transfer_ready[1]},
        .consumer_complete = {events.consumer_complete[0], events.consumer_complete[1]},
    };
    ops::qwen4_sparse_moe(x, weights, pipeline, selected_ids, selected_weights, destination,
                          moe_workspace, device.stream);
    device.synchronize();
    CUDA_CHECK(cudaStreamSynchronize(events.transfer_stream));

    int failures = 0;
    const std::vector<std::int32_t> actual_ids =
        from_device<std::int32_t>(device_ids.data(), kTopK);
    const std::vector<std::int32_t> expected_ids(route.ids.begin(), route.ids.end());
    if (actual_ids != expected_ids) {
        std::cerr << "Qwen4 real layer-0 sparse-MoE selected wrong expert ids\n";
        ++failures;
    }
    const std::vector<float> actual_weight_f32 =
        from_device<float>(device_weights.data(), kTopK);
    const std::vector<double> actual_weights(actual_weight_f32.begin(),
                                             actual_weight_f32.end());
    failures += verify_pointwise("Qwen4 real layer-0 sparse-MoE selected weights",
                                 actual_weights, route.weights, kRouteWeightCriterion);
    failures += verify_reduction("Qwen4 real layer-0 sparse-MoE complete FP64 formula",
                                 from_device_bf16(device_destination.data(), kHidden), reference,
                                 kOutputCriterion);

    const std::size_t routed_matrix =
        matrix_bytes(QType::GGML_IQ1_S, kIntermediate, kHidden);
    const std::size_t rank_bytes = 2 * routed_matrix;
    if (rank_bytes != ops::qwen4_sparse_moe_rank_stage_bytes(QType::GGML_IQ1_S)) {
        throw std::logic_error("Qwen4 layer-0 sparse-MoE rank-stage geometry changed");
    }
    std::vector<std::uint8_t> actual_stage(ops::kQwen4SparseMoePipelineStageBytes);
    device_stage.copy_to_host(actual_stage.data(), actual_stage.size());
    for (std::int32_t slot = 0; slot < ops::kQwen4SparseMoePipelineSlots; ++slot) {
        const std::int32_t rank = kTopK - ops::kQwen4SparseMoePipelineSlots + slot;
        const std::int32_t expert = route.ids[static_cast<std::size_t>(rank)];
        const std::span<const std::uint8_t> gate =
            expert_span(weights.routed_gate_up.gate, expert, routed_matrix);
        const std::span<const std::uint8_t> up =
            expert_span(weights.routed_gate_up.up, expert, routed_matrix);
        const std::size_t slot_offset = static_cast<std::size_t>(slot) *
                                        ops::kQwen4SparseMoeRankStageCapacityBytes;
        const auto exact_slot = [&](const std::uint8_t* staged) {
            return std::memcmp(staged, gate.data(), routed_matrix) == 0 &&
                   std::memcmp(staged + routed_matrix, up.data(), routed_matrix) == 0;
        };
        if (!exact_slot(static_cast<const std::uint8_t*>(pinned_stage) + slot_offset) ||
            !exact_slot(actual_stage.data() + slot_offset)) {
            std::cerr << "Qwen4 real layer-0 sparse-MoE surviving stage bytes changed\n";
            ++failures;
        }
        if (!std::all_of(pinned_bytes + kHostGuardBytes + slot_offset + rank_bytes,
                         pinned_bytes + kHostGuardBytes + slot_offset +
                             ops::kQwen4SparseMoeRankStageCapacityBytes,
                         [](std::uint8_t value) { return value == 0xa5; }) ||
            !std::all_of(actual_stage.begin() + static_cast<std::ptrdiff_t>(slot_offset +
                                                                           rank_bytes),
                         actual_stage.begin() + static_cast<std::ptrdiff_t>(
                             slot_offset + ops::kQwen4SparseMoeRankStageCapacityBytes),
                         [](std::uint8_t value) { return value == 0xcd; })) {
            std::cerr << "Qwen4 real layer-0 sparse-MoE copied beyond one rank stage\n";
            ++failures;
        }
    }
    if (!std::all_of(pinned_bytes, pinned_bytes + kHostGuardBytes,
                     [](std::uint8_t value) { return value == 0xa5; }) ||
        !std::all_of(pinned_bytes + kHostGuardBytes + ops::kQwen4SparseMoePipelineStageBytes,
                     pinned_bytes + pinned.size(),
                     [](std::uint8_t value) { return value == 0xa5; })) {
        std::cerr << "Qwen4 real layer-0 sparse-MoE pinned guard changed\n";
        ++failures;
    }
    failures += device_x.verify_guards("Qwen4 real layer-0 FFN GR x");
    failures += device_stage.verify_guards("Qwen4 real layer-0 sparse-MoE stage");
    failures += device_ids.verify_guards("Qwen4 real layer-0 sparse-MoE ids");
    failures += device_weights.verify_guards("Qwen4 real layer-0 sparse-MoE weights");
    failures += device_destination.verify_guards("Qwen4 real layer-0 sparse-MoE destination");
    failures += run_accumulated_layer35_cell(model, device);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4_real_moe_oracle_cell ids=";
    for (std::size_t rank = 0; rank < route.ids.size(); ++rank) {
        std::cout << (rank == 0 ? "" : ",") << route.ids[rank];
    }
    std::cout << '\n';
    return failures;
}

} // namespace ninfer::test::qwen4::real_oracle
