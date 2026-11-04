// Native wave32 gfx1201 implementation of the complete Qwen3.8 RoPE public Op boundary.
#include "ninfer/ops/rope.h"
#include "ninfer/ops/normalized_rope_kv_append.h"

#include "core/device.h"
#include "ninfer/types.h"
#include "ops/r9700/rope/rope_kernels.h"

#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using namespace rope_kernels;

template <int HeadDim, int Half>
__device__ __forceinline__ void apply_head(hip_bfloat16* data, std::int64_t token_stride, int token,
                                           int head, int lane, const float* sine,
                                           const float* cosine) {
    const std::int64_t base =
        static_cast<std::int64_t>(token) * token_stride + static_cast<std::int64_t>(head) * HeadDim;
    for (int pair = lane; pair < Half; pair += 32) {
        const float first        = static_cast<float>(data[base + pair]);
        const float second       = static_cast<float>(data[base + pair + Half]);
        data[base + pair]        = hip_bfloat16(first * cosine[pair] - second * sine[pair]);
        data[base + pair + Half] = hip_bfloat16(second * cosine[pair] + first * sine[pair]);
    }
}

template <Profile P, int HeadDim, int Half, int QHeads, int KHeads, bool Split>
__global__ __launch_bounds__(Split ? kSplitBlock : kCombinedBlock) void rope_kernel(
    const std::int32_t* positions, hip_bfloat16* q, hip_bfloat16* k, int tokens,
    std::int64_t q_stride, std::int64_t k_stride) {
    constexpr int combined_heads = QHeads + KHeads;
    constexpr int groups         = (combined_heads + kSplitHeads - 1) / kSplitHeads;
    const int token = Split ? static_cast<int>(blockIdx.x) / groups : static_cast<int>(blockIdx.x);
    const int group = Split ? static_cast<int>(blockIdx.x) % groups : 0;
    if (token >= tokens) { return; }

    __shared__ float sine[Half];
    __shared__ float cosine[Half];
    for (int pair = static_cast<int>(threadIdx.x); pair < Half; pair += blockDim.x) {
        coefficient<P>(positions, tokens, token, pair, &sine[pair], &cosine[pair]);
    }
    __syncthreads();

    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int wave      = static_cast<int>(threadIdx.x) >> 5;
    constexpr int waves = Split ? kSplitHeads : kCombinedBlock / 32;
    int combined        = Split ? group * kSplitHeads + wave : wave;
    for (; combined < combined_heads && (!Split || combined < (group + 1) * kSplitHeads);
         combined += waves) {
        if (combined < QHeads) {
            apply_head<HeadDim, Half>(q, q_stride, token, combined, lane, sine, cosine);
        } else {
            apply_head<HeadDim, Half>(k, k_stride, token, combined - QHeads, lane, sine, cosine);
        }
    }
}

// Each lane owns four adjacent features, including their opposite-half partners.
// Load both halves before the barrier: stores may otherwise overtake another lane's partner.
constexpr int kVisionVectorBlock = 128;

__global__ __launch_bounds__(kVisionVectorBlock) void vision_rope_vector_kernel(
    const std::int32_t* positions, hip_bfloat16* q, hip_bfloat16* k, int tokens,
    std::int64_t q_stride, std::int64_t k_stride) {
    constexpr int vectors = 2 * kVisionH * kVisionD / 4;
    constexpr int items   = (vectors + kVisionVectorBlock - 1) / kVisionVectorBlock;
    const int token       = static_cast<int>(blockIdx.x);
    __shared__ float sine[36];
    __shared__ float cosine[36];
    if (threadIdx.x < 36) {
        coefficient<Profile::Vision2D>(positions, tokens, token, static_cast<int>(threadIdx.x),
                                       &sine[threadIdx.x], &cosine[threadIdx.x]);
    }
    uint2 original[items], partner[items];
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int index = static_cast<int>(threadIdx.x) + i * kVisionVectorBlock;
        if (index < vectors) {
            const int head    = index / 18;
            const int feature = index % 18 * 4;
            const auto* data =
                head < kVisionH
                    ? q + static_cast<std::int64_t>(token) * q_stride + head * 72
                    : k + static_cast<std::int64_t>(token) * k_stride + (head - kVisionH) * 72;
            original[i] = *reinterpret_cast<const uint2*>(data + feature);
            partner[i]  = *reinterpret_cast<const uint2*>(
                data + (feature < 36 ? feature + 36 : feature - 36));
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int index = static_cast<int>(threadIdx.x) + i * kVisionVectorBlock;
        if (index < vectors) {
            const int head    = index / 18;
            const int feature = index % 18 * 4;
            auto* data =
                head < kVisionH
                    ? q + static_cast<std::int64_t>(token) * q_stride + head * 72
                    : k + static_cast<std::int64_t>(token) * k_stride + (head - kVisionH) * 72;
            const std::uint32_t own[2]      = {original[i].x, original[i].y};
            const std::uint32_t opposite[2] = {partner[i].x, partner[i].y};
            std::uint32_t words[2]          = {};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                hip_bfloat16 a, b;
                a.data             = static_cast<std::uint16_t>(own[j / 2] >> (16 * (j % 2)));
                b.data             = static_cast<std::uint16_t>(opposite[j / 2] >> (16 * (j % 2)));
                const int pair     = (feature + j) % 36;
                const float first  = static_cast<float>(a);
                const float second = static_cast<float>(b);
                const auto value =
                    hip_bfloat16(feature < 36 ? first * cosine[pair] - second * sine[pair]
                                              : first * cosine[pair] + second * sine[pair]);
                words[j / 2] |= static_cast<std::uint32_t>(value.data) << (16 * (j % 2));
            }
            *reinterpret_cast<uint2*>(data + feature) = {words[0], words[1]};
        }
    }
}

// DFlash drafter front: one wave per (token, head) of the fused [Q|K|V] rows, four features per
// lane (feature 4 * lane + i). Normalized heads sum their squares serially in feature order in
// every lane (broadcast loads), exactly as the eager RMSNorm's single summing thread; RoPE pairs
// feature f < 64 in lane l with f + 64 in lane l + 16.
constexpr int kDflashFusedRows  = (kDflashQ + 2 * kDflashK) * kDflashD;
constexpr int kDflashFrontWaves = 4;

template <bool Append>
__global__ __launch_bounds__(kDflashFrontWaves * 32) void dflash_qkv_norm_rope_kernel(
    const std::int32_t* positions, const hip_bfloat16* fused, const hip_bfloat16* q_norm,
    const hip_bfloat16* k_norm, float eps, hip_bfloat16* qn, hip_bfloat16* kn, hip_bfloat16* v,
    int tokens, int first_head, const std::int32_t* counts, const std::int32_t* lanes, int width,
    int capacity) {
    constexpr int heads = (Append ? 0 : kDflashQ) + 2 * kDflashK;
    const int per_token = heads - first_head;
    const int groups    = (per_token + kDflashFrontWaves - 1) / kDflashFrontWaves;
    const int token     = static_cast<int>(blockIdx.x) / groups;
    const int head      = first_head + (static_cast<int>(blockIdx.x) % groups) * kDflashFrontWaves +
                          static_cast<int>(threadIdx.x) / 32;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    if (token >= tokens || head >= heads) { return; }
    if constexpr (Append) {
        if (token % width >= counts[token / width]) return;
    }
    const auto kv_offset = [&](int kv_head) {
        if constexpr (Append) {
            return ((static_cast<std::int64_t>(lanes[token / width]) * kDflashK + kv_head) *
                        capacity +
                    positions[token] % capacity) *
                   kDflashD;
        } else {
            return (static_cast<std::int64_t>(token) * kDflashK + kv_head) * kDflashD;
        }
    };
    constexpr int query_heads = Append ? 0 : kDflashQ;
    constexpr int fused_rows  = heads * kDflashD;
    const hip_bfloat16* row   = fused + static_cast<std::int64_t>(token) * fused_rows +
                                static_cast<std::int64_t>(head) * kDflashD;
    const uint2 packed        = reinterpret_cast<const uint2*>(row)[lane];
    if (head >= query_heads + kDflashK) {
        reinterpret_cast<uint2*>(v + kv_offset(head - query_heads - kDflashK))[lane] = packed;
        return;
    }
    const bool is_query = head < query_heads;
    const auto element  = [](std::uint32_t word, bool high) {
        return __uint_as_float(high ? word & 0xffff0000U : word << 16);
    };
    float sumsq         = 0.0F;
    const auto* vectors = reinterpret_cast<const uint4*>(row);
    for (int vector = 0; vector < kDflashD / 8; ++vector) {
        const uint4 all              = vectors[vector];
        const std::uint32_t words[4] = {all.x, all.y, all.z, all.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float x = element(words[i >> 1], (i & 1) != 0);
            sumsq         = fmaf(x, x, sumsq);
        }
    }
    const float inverse          = rsqrtf(sumsq / static_cast<float>(kDflashD) + eps);
    const uint2 gains            = reinterpret_cast<const uint2*>(is_query ? q_norm : k_norm)[lane];
    const std::uint32_t words[2] = {packed.x, packed.y};
    const std::uint32_t gain_words[2] = {gains.x, gains.y};
    float value[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float gain = element(gain_words[i >> 1], (i & 1) != 0);
        value[i] =
            static_cast<float>(hip_bfloat16(element(words[i >> 1], (i & 1) != 0) * inverse * gain));
    }
    float partner[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) partner[i] = __shfl_xor(value[i], 16, 32);
    std::uint16_t out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int feature = lane * 4 + i;
        const int pair    = feature % (kDflashD / 2);
        float sine = 0.0F, cosine = 0.0F;
        coefficient<Profile::Dflash1D>(positions, tokens, token, pair, &sine, &cosine);
        out[i] = feature < kDflashD / 2 ? hip_bfloat16(value[i] * cosine - partner[i] * sine).data
                                        : hip_bfloat16(value[i] * cosine + partner[i] * sine).data;
    }
    hip_bfloat16* destination =
        is_query ? qn + (static_cast<std::int64_t>(token) * kDflashQ + head) * kDflashD
                 : kn + kv_offset(head - query_heads);
    reinterpret_cast<uint2*>(destination)[lane] =
        uint2{static_cast<std::uint32_t>(out[0]) | (static_cast<std::uint32_t>(out[1]) << 16),
              static_cast<std::uint32_t>(out[2]) | (static_cast<std::uint32_t>(out[3]) << 16)};
}

template <Profile P, int HeadDim, int Half, int QHeads, int KHeads>
void launch_profile(const std::int32_t* positions, hip_bfloat16* q, hip_bfloat16* k, int tokens,
                    std::int64_t q_stride, std::int64_t k_stride, hipStream_t stream) {
    if (tokens <= 8) {
        constexpr int groups = (QHeads + KHeads + kSplitHeads - 1) / kSplitHeads;
        rope_kernel<P, HeadDim, Half, QHeads, KHeads, true>
            <<<tokens * groups, kSplitBlock, 0, stream>>>(positions, q, k, tokens, q_stride,
                                                          k_stride);
    } else {
        rope_kernel<P, HeadDim, Half, QHeads, KHeads, false>
            <<<tokens, kCombinedBlock, 0, stream>>>(positions, q, k, tokens, q_stride, k_stride);
    }
}

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    bool zero          = false;
    std::int64_t count = 1;
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] < 0) {
            throw std::invalid_argument(std::string("rope: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (tensor.ne[dim] == 0) {
            zero = true;
            continue;
        }
        if (count > std::numeric_limits<std::int64_t>::max() / tensor.ne[dim]) {
            throw std::overflow_error("rope: tensor size overflows int64");
        }
        count *= tensor.ne[dim];
    }
    return zero ? 0 : count;
}

int axes_for(const Tensor& positions, int tokens) {
    if (positions.ne[0] != tokens || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        (positions.ne[1] != 1 && positions.ne[1] != 2 && positions.ne[1] != 3)) {
        throw std::invalid_argument("rope: positions must have shape [T], [T,2], or [T,3]");
    }
    return positions.ne[1];
}

void require_layout(const Tensor& tensor, const char* label, int head_dim, int heads, int tokens) {
    if (tensor.ne[0] != head_dim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " shape");
    }
    constexpr std::int64_t elem = sizeof(hip_bfloat16);
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * static_cast<std::int64_t>(head_dim) * heads ||
        tensor.nb[2] % elem != 0) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " strides");
    }
}

Profile pair_profile(int axes, int rotary_dim, float theta, const Tensor& q, const Tensor& k) {
    if ((axes == 1 || axes == 3) && rotary_dim == kTextR && theta == 1.0e7F && q.ne[0] == kTextD &&
        q.ne[1] == kTextQ && k.ne[0] == kTextD && k.ne[1] == kTextK) {
        return axes == 1 ? Profile::Text1D : Profile::TextMrope;
    }
    if (axes == 1 && rotary_dim == kDflashD && theta == 1.0e7F && q.ne[0] == kDflashD &&
        q.ne[1] == kDflashQ && k.ne[0] == kDflashD && k.ne[1] == kDflashK) {
        return Profile::Dflash1D;
    }
    if (axes == 2 && rotary_dim == kVisionD && theta == 10'000.0F && q.ne[0] == kVisionD &&
        q.ne[1] == kVisionH && k.ne[0] == kVisionD && k.ne[1] == kVisionH) {
        return Profile::Vision2D;
    }
    throw std::invalid_argument("rope: unsupported Qwen3.8 pair geometry");
}

Profile single_profile(int axes, int rotary_dim, float theta, const Tensor& x) {
    if ((axes == 1 || axes == 3) && rotary_dim == kTextR && theta == 1.0e7F && x.ne[0] == kTextD &&
        (x.ne[1] == kTextQ || x.ne[1] == kTextK)) {
        return axes == 1 ? Profile::Text1D : Profile::TextMrope;
    }
    if (axes == 1 && rotary_dim == kDflashD && theta == 1.0e7F && x.ne[0] == kDflashD &&
        (x.ne[1] == kDflashQ || x.ne[1] == kDflashK)) {
        return Profile::Dflash1D;
    }
    if (axes == 2 && rotary_dim == kVisionD && theta == 10'000.0F && x.ne[0] == kVisionD &&
        x.ne[1] == kVisionH) {
        return Profile::Vision2D;
    }
    throw std::invalid_argument("rope: unsupported Qwen3.8 single-tensor geometry");
}

std::size_t storage_bytes(const Tensor& tensor) {
    if (tensor.ne[2] == 0) { return 0; }
    return static_cast<std::size_t>((tensor.ne[2] - 1) * tensor.nb[2] +
                                    tensor.ne[1] * tensor.nb[1]);
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto aa = reinterpret_cast<std::uintptr_t>(a);
    const auto bb = reinterpret_cast<std::uintptr_t>(b);
    return a_bytes != 0 && b_bytes != 0 && aa < bb + b_bytes && bb < aa + a_bytes;
}

bool tensor_payloads_overlap(const Tensor& a, const Tensor& b) {
    if (!overlaps(a.data, storage_bytes(a), b.data, storage_bytes(b))) { return false; }
    // Q/K may be disjoint planes of token-interleaved QKV. Their bounding spans include
    // padding owned by the other planes, which this Op never reads or writes.
    const auto a_width = static_cast<std::uintptr_t>(a.ne[1] * a.nb[1]);
    const auto b_width = static_cast<std::uintptr_t>(b.ne[1] * b.nb[1]);
    auto aa            = reinterpret_cast<std::uintptr_t>(a.data);
    auto bb            = reinterpret_cast<std::uintptr_t>(b.data);
    std::int32_t ai = 0, bi = 0;
    while (ai < a.ne[2] && bi < b.ne[2]) {
        if (aa < bb + b_width && bb < aa + a_width) { return true; }
        if (aa + a_width <= bb) {
            ++ai;
            aa += a.nb[2];
        } else {
            ++bi;
            bb += b.nb[2];
        }
    }
    return false;
}

void require_common(const Tensor& positions, int rotary_dim, float theta) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope: positions must be I32");
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || !(theta > 0.0F) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: invalid rotary_dim or theta");
    }
    (void)checked_numel(positions, "positions");
}

void launch(Profile profile, const Tensor& positions, Tensor& q, Tensor* k, hipStream_t stream) {
    auto* q_data     = static_cast<hip_bfloat16*>(q.data);
    auto* k_data     = k == nullptr ? nullptr : static_cast<hip_bfloat16*>(k->data);
    const auto* pos  = static_cast<const std::int32_t*>(positions.data);
    const int tokens = q.ne[2];
    const auto qs    = q.nb[2] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
    const auto ks = k == nullptr ? 0 : k->nb[2] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
    switch (profile) {
    case Profile::Text1D:
        if (k != nullptr)
            launch_profile<Profile::Text1D, 256, 32, 24, 4>(pos, q_data, k_data, tokens, qs, ks,
                                                            stream);
        else if (q.ne[1] == 24)
            launch_profile<Profile::Text1D, 256, 32, 24, 0>(pos, q_data, nullptr, tokens, qs, 0,
                                                            stream);
        else
            launch_profile<Profile::Text1D, 256, 32, 4, 0>(pos, q_data, nullptr, tokens, qs, 0,
                                                           stream);
        break;
    case Profile::TextMrope:
        if (k != nullptr)
            launch_profile<Profile::TextMrope, 256, 32, 24, 4>(pos, q_data, k_data, tokens, qs, ks,
                                                               stream);
        else if (q.ne[1] == 24)
            launch_profile<Profile::TextMrope, 256, 32, 24, 0>(pos, q_data, nullptr, tokens, qs, 0,
                                                               stream);
        else
            launch_profile<Profile::TextMrope, 256, 32, 4, 0>(pos, q_data, nullptr, tokens, qs, 0,
                                                              stream);
        break;
    case Profile::Dflash1D:
        if (k != nullptr)
            launch_profile<Profile::Dflash1D, 128, 64, 32, 8>(pos, q_data, k_data, tokens, qs, ks,
                                                              stream);
        else if (q.ne[1] == 32)
            launch_profile<Profile::Dflash1D, 128, 64, 32, 0>(pos, q_data, nullptr, tokens, qs, 0,
                                                              stream);
        else
            launch_profile<Profile::Dflash1D, 128, 64, 8, 0>(pos, q_data, nullptr, tokens, qs, 0,
                                                             stream);
        break;
    case Profile::Vision2D:
        if (k != nullptr && tokens >= 256 && reinterpret_cast<std::uintptr_t>(q_data) % 8U == 0U &&
            reinterpret_cast<std::uintptr_t>(k_data) % 8U == 0U && qs % 4 == 0 && ks % 4 == 0) {
            vision_rope_vector_kernel<<<tokens, kVisionVectorBlock, 0, stream>>>(
                pos, q_data, k_data, tokens, qs, ks);
        } else if (k != nullptr) {
            launch_profile<Profile::Vision2D, 72, 36, 16, 16>(pos, q_data, k_data, tokens, qs, ks,
                                                              stream);
        } else
            launch_profile<Profile::Vision2D, 72, 36, 16, 0>(pos, q_data, nullptr, tokens, qs, 0,
                                                             stream);
        break;
    }
    HIP_CHECK(hipGetLastError());
}

} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          hipStream_t stream) {
    require_common(positions, rotary_dim, theta);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    const auto q_count = checked_numel(q, "q");
    (void)checked_numel(k, "k");
    const int tokens      = q.ne[2];
    const int axes        = axes_for(positions, tokens);
    const Profile profile = pair_profile(axes, rotary_dim, theta, q, k);
    require_layout(q, "q", q.ne[0], q.ne[1], tokens);
    require_layout(k, "k", k.ne[0], k.ne[1], tokens);
    if (q_count == 0) { return; }
    if (positions.data == nullptr || q.data == nullptr || k.data == nullptr ||
        !positions.is_contiguous()) {
        throw std::invalid_argument("rope: storage must be non-null and positions contiguous");
    }
    const std::size_t p_bytes = positions.bytes();
    const std::size_t q_bytes = storage_bytes(q);
    const std::size_t k_bytes = storage_bytes(k);
    if (tensor_payloads_overlap(q, k) || overlaps(positions.data, p_bytes, q.data, q_bytes) ||
        overlaps(positions.data, p_bytes, k.data, k_bytes)) {
        throw std::invalid_argument("rope: positions, q, and k storage must not overlap");
    }
    launch(profile, positions, q, &k, stream);
}

void qk_norm_rope(const Tensor& positions, int rotary_dim, float theta, const Tensor& q,
                  const Tensor& k, const Tensor& q_norm, const Tensor& k_norm, float eps,
                  Tensor& qn, Tensor& kn, hipStream_t stream) {
    require_common(positions, rotary_dim, theta);
    const int tokens      = q.ne[2];
    const int axes        = axes_for(positions, tokens);
    const Profile profile = pair_profile(axes, rotary_dim, theta, q, k);
    if (profile != Profile::Text1D && profile != Profile::TextMrope) {
        throw std::invalid_argument("qk_norm_rope: only the Qwen3.8 Text geometry is supported");
    }
    const auto bf16_vector = [](const Tensor& t, int n) {
        return t.dtype == DType::BF16 && t.data != nullptr && t.is_contiguous() && t.ne[0] == n &&
               t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1 &&
               reinterpret_cast<std::uintptr_t>(t.data) % 16U == 0U;
    };
    const auto aligned_rows = [](const Tensor& t) {
        return t.dtype == DType::BF16 && t.data != nullptr &&
               reinterpret_cast<std::uintptr_t>(t.data) % 16U == 0U && t.nb[2] % 16 == 0;
    };
    if (!(eps > 0.0F) || !std::isfinite(eps) || !bf16_vector(q_norm, kTextD) ||
        !bf16_vector(k_norm, kTextD) || !aligned_rows(q) || !aligned_rows(k) || k.ne[2] != tokens ||
        qn.dtype != DType::BF16 || kn.dtype != DType::BF16 || !qn.is_contiguous() ||
        !kn.is_contiguous() || qn.ne[0] != kTextD || qn.ne[1] != kTextQ || qn.ne[2] != tokens ||
        kn.ne[0] != kTextD || kn.ne[1] != kTextK || kn.ne[2] != tokens ||
        reinterpret_cast<std::uintptr_t>(qn.data) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(kn.data) % 16U != 0U) {
        throw std::invalid_argument("qk_norm_rope: invalid tensors");
    }
    require_layout(q, "q", kTextD, kTextQ, tokens);
    require_layout(k, "k", kTextD, kTextK, tokens);
    if (tokens == 0) { return; }
    const std::size_t p_bytes    = positions.bytes();
    const std::size_t outputs[2] = {qn.bytes(), kn.bytes()};
    const void* output_data[2]   = {qn.data, kn.data};
    for (int i = 0; i < 2; ++i) {
        if (overlaps(output_data[i], outputs[i], q.data, storage_bytes(q)) ||
            overlaps(output_data[i], outputs[i], k.data, storage_bytes(k)) ||
            overlaps(output_data[i], outputs[i], positions.data, p_bytes)) {
            throw std::invalid_argument("qk_norm_rope: outputs must not overlap inputs");
        }
    }
    if (overlaps(qn.data, outputs[0], kn.data, outputs[1])) {
        throw std::invalid_argument("qk_norm_rope: outputs must not overlap");
    }
    const auto qs = q.nb[2] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
    const auto ks = k.nb[2] / static_cast<std::int64_t>(sizeof(hip_bfloat16));
    const dim3 grid(static_cast<unsigned>(tokens * kQkNormRopeGroups));
    const auto* pos    = static_cast<const std::int32_t*>(positions.data);
    const auto* q_data = static_cast<const hip_bfloat16*>(q.data);
    const auto* k_data = static_cast<const hip_bfloat16*>(k.data);
    const auto* qw     = static_cast<const hip_bfloat16*>(q_norm.data);
    const auto* kw     = static_cast<const hip_bfloat16*>(k_norm.data);
    auto* qo           = static_cast<hip_bfloat16*>(qn.data);
    auto* ko           = static_cast<hip_bfloat16*>(kn.data);
    if (profile == Profile::Text1D) {
        qk_norm_rope_kernel<Profile::Text1D><<<grid, kQkNormRopeWaves * 32, 0, stream>>>(
            pos, q_data, k_data, qs, ks, qw, kw, eps, qo, ko, tokens);
    } else {
        qk_norm_rope_kernel<Profile::TextMrope><<<grid, kQkNormRopeWaves * 32, 0, stream>>>(
            pos, q_data, k_data, qs, ks, qw, kw, eps, qo, ko, tokens);
    }
    HIP_CHECK(hipGetLastError());
}

void dflash_qkv_norm_rope(const Tensor& positions, const Tensor& fused, const Tensor& q_norm,
                          const Tensor& k_norm, float eps, Tensor* qn, Tensor& kn, Tensor& v,
                          hipStream_t stream) {
    const int tokens     = fused.ne[1];
    const auto bf16_rows = [&](const Tensor& t, int rows, int columns) {
        return t.dtype == DType::BF16 && t.data != nullptr && t.is_contiguous() &&
               t.numel() == static_cast<std::int64_t>(rows) * columns &&
               reinterpret_cast<std::uintptr_t>(t.data) % 16U == 0U;
    };
    const auto heads = [&](const Tensor& t, int count) {
        return bf16_rows(t, kDflashD * count, tokens) && t.ne[0] == kDflashD && t.ne[1] == count &&
               t.ne[2] == tokens && t.ne[3] == 1;
    };
    if (tokens <= 0 || fused.ne[0] != kDflashFusedRows || fused.ne[2] != 1 || fused.ne[3] != 1 ||
        !bf16_rows(fused, kDflashFusedRows, tokens) || !bf16_rows(q_norm, kDflashD, 1) ||
        !bf16_rows(k_norm, kDflashD, 1) || !(eps > 0.0F) || !std::isfinite(eps) ||
        positions.dtype != DType::I32 || positions.data == nullptr || !positions.is_contiguous() ||
        positions.numel() != tokens || !heads(kn, kDflashK) || !heads(v, kDflashK) ||
        (qn != nullptr && !heads(*qn, kDflashQ))) {
        throw std::invalid_argument("dflash_qkv_norm_rope: invalid tensors");
    }
    const std::size_t fused_bytes = fused.bytes();
    for (const Tensor* output : {static_cast<const Tensor*>(qn), static_cast<const Tensor*>(&kn),
                                 static_cast<const Tensor*>(&v)}) {
        if (output == nullptr) continue;
        if (overlaps(output->data, output->bytes(), fused.data, fused_bytes) ||
            overlaps(output->data, output->bytes(), positions.data, positions.bytes())) {
            throw std::invalid_argument("dflash_qkv_norm_rope: outputs must not overlap inputs");
        }
    }
    if (overlaps(kn.data, kn.bytes(), v.data, v.bytes()) ||
        (qn != nullptr && (overlaps(qn->data, qn->bytes(), kn.data, kn.bytes()) ||
                           overlaps(qn->data, qn->bytes(), v.data, v.bytes())))) {
        throw std::invalid_argument("dflash_qkv_norm_rope: outputs must not overlap");
    }
    const int first_head = qn != nullptr ? 0 : kDflashQ;
    const int per_token  = kDflashQ + 2 * kDflashK - first_head;
    const int groups     = (per_token + kDflashFrontWaves - 1) / kDflashFrontWaves;
    dflash_qkv_norm_rope_kernel<false><<<tokens * groups, kDflashFrontWaves * 32, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const hip_bfloat16*>(fused.data), static_cast<const hip_bfloat16*>(q_norm.data),
        static_cast<const hip_bfloat16*>(k_norm.data), eps,
        qn != nullptr ? static_cast<hip_bfloat16*>(qn->data) : nullptr,
        static_cast<hip_bfloat16*>(kn.data), static_cast<hip_bfloat16*>(v.data), tokens, first_head,
        nullptr, nullptr, 0, 0);
    HIP_CHECK(hipGetLastError());
}

void normalized_rope_kv_append(const Tensor& fused_kv, const Tensor& key_norm,
                               const Tensor& positions, const Tensor& counts, const Tensor& lanes,
                               float eps, KVCacheAppendPrefixExecutionEnvelope envelope,
                               const CyclicKVCacheLayerView& cache, hipStream_t stream) {
    const int width = positions.ne[0], batch = positions.ne[1];
    const auto valid = [](const Tensor& tensor, DType type, int x, int y, int z, int w) {
        return tensor.dtype == type && tensor.data != nullptr && tensor.is_contiguous() &&
               tensor.ne[0] == x && tensor.ne[1] == y && tensor.ne[2] == z && tensor.ne[3] == w &&
               reinterpret_cast<std::uintptr_t>(tensor.data) % (type == DType::I32 ? 4U : 16U) ==
                   0U;
    };
    if (width <= 0 || width > 2048 || batch < 1 ||
        batch > static_cast<int>(kDFlashMaximumConcurrency) || cache.capacity != 2048 ||
        cache.padded_capacity != 2048 || cache.head_dim != 128 || cache.num_kv_heads != 8 ||
        cache.lane_capacity < batch ||
        cache.lane_capacity > static_cast<int>(kDFlashMaximumConcurrency) ||
        envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<unsigned>(width) || !(eps > 0.0F) || !std::isfinite(eps) ||
        !valid(fused_kv, DType::BF16, 2048, width * batch, 1, 1) ||
        !valid(key_norm, DType::BF16, 128, 1, 1, 1) ||
        !valid(positions, DType::I32, width, batch, 1, 1) ||
        !valid(counts, DType::I32, batch, 1, 1, 1) || !valid(lanes, DType::I32, batch, 1, 1, 1) ||
        !valid(cache.k, DType::BF16, 128, 2048, 8, cache.lane_capacity) ||
        !valid(cache.v, DType::BF16, 128, 2048, 8, cache.lane_capacity)) {
        throw std::invalid_argument("normalized_rope_kv_append: invalid tensors or envelope");
    }
    for (const Tensor* input : {&fused_kv, &key_norm, &positions, &counts, &lanes}) {
        if (overlaps(input->data, input->bytes(), cache.k.data, cache.k.bytes()) ||
            overlaps(input->data, input->bytes(), cache.v.data, cache.v.bytes()))
            throw std::invalid_argument("normalized_rope_kv_append: cache aliases input");
    }
    if (overlaps(cache.k.data, cache.k.bytes(), cache.v.data, cache.v.bytes()))
        throw std::invalid_argument("normalized_rope_kv_append: cache planes overlap");
    if (envelope.max_count == 0) return;
    dflash_qkv_norm_rope_kernel<true><<<width * batch * 4, kDflashFrontWaves * 32, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const hip_bfloat16*>(fused_kv.data), nullptr,
        static_cast<const hip_bfloat16*>(key_norm.data), eps, nullptr,
        static_cast<hip_bfloat16*>(cache.k.data), static_cast<hip_bfloat16*>(cache.v.data),
        width * batch, 0, static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(lanes.data), width, 2048);
    HIP_CHECK(hipGetLastError());
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, hipStream_t stream) {
    require_common(positions, rotary_dim, theta);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    const auto count      = checked_numel(x, "tensor");
    const int tokens      = x.ne[2];
    const int axes        = axes_for(positions, tokens);
    const Profile profile = single_profile(axes, rotary_dim, theta, x);
    require_layout(x, "tensor", x.ne[0], x.ne[1], tokens);
    if (count == 0) { return; }
    if (positions.data == nullptr || x.data == nullptr || !positions.is_contiguous()) {
        throw std::invalid_argument("rope: storage must be non-null and positions contiguous");
    }
    if (overlaps(positions.data, positions.bytes(), x.data, storage_bytes(x))) {
        throw std::invalid_argument("rope: positions and tensor storage must not overlap");
    }
    launch(profile, positions, x, nullptr, stream);
}

} // namespace ninfer::ops
