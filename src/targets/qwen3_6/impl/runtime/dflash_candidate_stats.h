#pragma once

// Opt-in DFlash2 miss-ceiling probe. NINFER_DFLASH_CANDIDATE_STATS=1 copies draft
// logits after propose (sync D2H) and classifies hops / the first reject token.
// Disabled when the env var is unset.

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

inline bool dflash_candidate_stats_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_DFLASH_CANDIDATE_STATS");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

namespace dflash_candidate_stats {
namespace {

constexpr int kMaxDrafts = 8;
constexpr int kMaxWidth  = 16;
constexpr int kVocabCap  = 248320;

struct Probe {
    std::mutex mu;
    int rows   = 0;
    int drafts = 0;
    std::vector<std::uint16_t> logits;
    std::vector<std::int32_t> token_ids;
    std::vector<std::int32_t> token_row;
    std::uint64_t hops                = 0;
    std::uint64_t hits                = 0;
    std::uint64_t in_tree             = 0;
    std::uint64_t in_top16            = 0;
    std::uint64_t in_top64            = 0;
    std::uint64_t in_top256           = 0;
    std::uint64_t in_draft_vocab      = 0;
    std::uint64_t missing_draft_vocab = 0;
    std::uint64_t depth_hops[kMaxDrafts]{};
    std::uint64_t depth_hits[kMaxDrafts]{};
    std::uint64_t depth_top16[kMaxDrafts]{};
    std::uint64_t depth_top256[kMaxDrafts]{};
    std::uint64_t rejects             = 0;
    std::uint64_t reject_in_tree      = 0;
    std::uint64_t reject_top16        = 0;
    std::uint64_t reject_top64        = 0;
    std::uint64_t reject_top256       = 0;
    std::uint64_t reject_in_head      = 0;
    std::uint64_t reject_absent_head  = 0;
    std::uint64_t reject_depth[kMaxDrafts]{};
    bool printed       = false;
    bool logged_health = false;
};

inline Probe& probe() {
    static Probe p;
    return p;
}

inline float bf16_f32(std::uint16_t bits) {
    const std::uint32_t s = static_cast<std::uint32_t>(bits) << 16;
    float v;
    std::memcpy(&v, &s, sizeof(v));
    return v;
}

inline void print_report(Probe& p) {
    if (p.printed || p.hops == 0) { return; }
    p.printed          = true;
    const auto pct     = [&](std::uint64_t n) {
        return 100.0 * static_cast<double>(n) / static_cast<double>(p.hops);
    };
    const auto rpct = [&](std::uint64_t n) {
        return p.rejects == 0
                   ? 0.0
                   : 100.0 * static_cast<double>(n) / static_cast<double>(p.rejects);
    };
    std::fprintf(stderr,
                 "dflash_candidate_stats hops=%llu hit=%.1f%% in_tree=%.1f%% top16=%.1f%% "
                 "top64=%.1f%% top256=%.1f%% in_draft_head=%.1f%% absent_from_head=%.1f%%\n",
                 static_cast<unsigned long long>(p.hops), pct(p.hits), pct(p.in_tree),
                 pct(p.in_top16), pct(p.in_top64), pct(p.in_top256), pct(p.in_draft_vocab),
                 pct(p.missing_draft_vocab));
    std::fprintf(stderr,
                 "dflash_candidate_stats REJECT n=%llu in_tree=%.1f%% top16=%.1f%% top64=%.1f%% "
                 "top256=%.1f%% in_head=%.1f%% absent_head=%.1f%%\n",
                 static_cast<unsigned long long>(p.rejects), rpct(p.reject_in_tree),
                 rpct(p.reject_top16), rpct(p.reject_top64), rpct(p.reject_top256),
                 rpct(p.reject_in_head), rpct(p.reject_absent_head));
    std::fprintf(stderr, "dflash_candidate_stats by_depth hops/hit/top16/top256:\n");
    for (int d = 0; d < p.drafts && d < kMaxDrafts; ++d) {
        if (p.depth_hops[d] == 0) { continue; }
        const double h = static_cast<double>(p.depth_hops[d]);
        std::fprintf(stderr, "  d%d n=%llu hit=%.1f%% top16=%.1f%% top256=%.1f%% reject_n=%llu\n",
                     d, static_cast<unsigned long long>(p.depth_hops[d]),
                     100.0 * static_cast<double>(p.depth_hits[d]) / h,
                     100.0 * static_cast<double>(p.depth_top16[d]) / h,
                     100.0 * static_cast<double>(p.depth_top256[d]) / h,
                     static_cast<unsigned long long>(p.reject_depth[d]));
    }
}

struct PrintOnExit {
    ~PrintOnExit() { print_report(probe()); }
};

inline PrintOnExit& printer() {
    static PrintOnExit p;
    return p;
}

inline int rank_in_column(const Probe& p, int token, int depth) {
    if (token < 0 || token >= static_cast<int>(p.token_row.size())) { return -1; }
    const int row = p.token_row[static_cast<std::size_t>(token)];
    if (row < 0) { return -1; }
    const float target =
        bf16_f32(p.logits[static_cast<std::size_t>(depth) * static_cast<std::size_t>(p.rows) +
                          static_cast<std::size_t>(row)]);
    int better               = 0;
    const std::uint16_t* col = p.logits.data() +
                               static_cast<std::size_t>(depth) * static_cast<std::size_t>(p.rows);
    for (int r = 0; r < p.rows; ++r) {
        const float v = bf16_f32(col[r]);
        if (v > target || (v == target && r < row)) { ++better; }
    }
    return better;
}

} // namespace

inline void capture_logits(const Tensor& logits, const Tensor* logit_token_ids, int drafts,
                           cudaStream_t stream) {
    if (!dflash_candidate_stats_enabled()) { return; }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) { return; }
    (void)printer();
    if (logits.dtype != DType::BF16) { return; }
    Probe& p = probe();
    std::lock_guard<std::mutex> lock(p.mu);
    p.rows   = logits.ne[0];
    p.drafts = drafts;
    const std::size_t n = static_cast<std::size_t>(p.rows) * static_cast<std::size_t>(drafts);
    p.logits.resize(n);
    CUDA_CHECK(cudaMemcpyAsync(p.logits.data(), logits.data, n * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    if (logit_token_ids != nullptr && p.token_ids.empty()) {
        p.token_ids.resize(static_cast<std::size_t>(p.rows));
        CUDA_CHECK(cudaMemcpyAsync(p.token_ids.data(), logit_token_ids->data,
                                   static_cast<std::size_t>(p.rows) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!p.logged_health) {
        p.logged_health = true;
        std::uint64_t nans = 0;
        std::uint64_t infs = 0;
        float mn           = std::numeric_limits<float>::infinity();
        float mx           = -std::numeric_limits<float>::infinity();
        for (std::uint16_t bits : p.logits) {
            const float v = bf16_f32(bits);
            if (!std::isfinite(v)) {
                if (std::isnan(v)) {
                    ++nans;
                } else {
                    ++infs;
                }
                continue;
            }
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        std::fprintf(stderr,
                     "dflash_candidate_stats logits rows=%d drafts=%d n=%zu nan=%llu inf=%llu "
                     "finite_min=%.4g finite_max=%.4g\n",
                     p.rows, p.drafts, p.logits.size(),
                     static_cast<unsigned long long>(nans),
                     static_cast<unsigned long long>(infs), mn, mx);
    }
    if (p.token_row.empty()) {
        p.token_row.assign(kVocabCap, -1);
        if (p.token_ids.empty()) {
            for (int r = 0; r < p.rows && r < kVocabCap; ++r) {
                p.token_row[static_cast<std::size_t>(r)] = r;
            }
        } else {
            for (int r = 0; r < p.rows; ++r) {
                const int tok = p.token_ids[static_cast<std::size_t>(r)];
                if (tok >= 0 && tok < kVocabCap) {
                    p.token_row[static_cast<std::size_t>(tok)] = r;
                }
            }
        }
    }
}

inline void capture_activation(const char* name, const Tensor& tensor, cudaStream_t stream) {
    if (!dflash_candidate_stats_enabled() || tensor.data == nullptr || tensor.dtype != DType::BF16) {
        return;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) { return; }
    const std::size_t n = std::min<std::size_t>(tensor.numel(), 65536);
    std::vector<std::uint16_t> bits(n);
    CUDA_CHECK(cudaMemcpyAsync(bits.data(), tensor.data, n * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::uint64_t nans = 0;
    std::uint64_t infs = 0;
    std::uint64_t zeros = 0;
    float mn            = std::numeric_limits<float>::infinity();
    float mx            = -std::numeric_limits<float>::infinity();
    float abs_sum       = 0.0f;
    for (std::uint16_t word : bits) {
        const float v = bf16_f32(word);
        if (!std::isfinite(v)) {
            if (std::isnan(v)) {
                ++nans;
            } else {
                ++infs;
            }
            continue;
        }
        if (v == 0.0f) { ++zeros; }
        abs_sum += std::fabs(v);
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    std::fprintf(stderr,
                 "dflash_candidate_stats %s numel=%lld sampled=%zu nan=%llu inf=%llu zero=%llu "
                 "mean_abs=%.4g finite_min=%.4g finite_max=%.4g\n",
                 name, static_cast<long long>(tensor.numel()), n,
                 static_cast<unsigned long long>(nans), static_cast<unsigned long long>(infs),
                 static_cast<unsigned long long>(zeros), abs_sum / static_cast<float>(n), mn, mx);
}

inline void record_round(const std::int32_t* verify_ids, const std::int32_t* parents,
                         const std::int32_t* licensed, int licensed_count, int width,
                         int drafts) {
    if (!dflash_candidate_stats_enabled() || licensed_count <= 0) { return; }
    Probe& p = probe();
    std::lock_guard<std::mutex> lock(p.mu);
    if (p.logits.empty() || p.rows <= 0) { return; }
    const int live_w = width < kMaxWidth ? width : kMaxWidth;
    int node         = 0;
    for (int hop = 0; hop < licensed_count; ++hop) {
        const int token = licensed[hop];
        const int depth = hop < drafts ? hop : drafts - 1;
        bool hit        = false;
        bool tree       = false;
        for (int c = 1; c < live_w; ++c) {
            if (verify_ids[c] == token) { tree = true; }
            if (parents[c] == node && verify_ids[c] == token) { hit = true; }
        }
        const int rank = rank_in_column(p, token, depth);
        ++p.hops;
        if (depth >= 0 && depth < kMaxDrafts) { ++p.depth_hops[depth]; }
        if (hit) {
            ++p.hits;
            if (depth >= 0 && depth < kMaxDrafts) { ++p.depth_hits[depth]; }
        }
        if (tree) { ++p.in_tree; }
        if (rank >= 0) {
            ++p.in_draft_vocab;
            if (rank < 16) {
                ++p.in_top16;
                if (depth >= 0 && depth < kMaxDrafts) { ++p.depth_top16[depth]; }
            }
            if (rank < 64) { ++p.in_top64; }
            if (rank < 256) {
                ++p.in_top256;
                if (depth >= 0 && depth < kMaxDrafts) { ++p.depth_top256[depth]; }
            }
        } else {
            ++p.missing_draft_vocab;
        }
        if (!hit) {
            ++p.rejects;
            if (tree) { ++p.reject_in_tree; }
            if (rank >= 0) {
                ++p.reject_in_head;
                if (rank < 16) { ++p.reject_top16; }
                if (rank < 64) { ++p.reject_top64; }
                if (rank < 256) { ++p.reject_top256; }
            } else {
                ++p.reject_absent_head;
            }
            if (depth >= 0 && depth < kMaxDrafts) { ++p.reject_depth[depth]; }
            break;
        }
        for (int c = 1; c < live_w; ++c) {
            if (parents[c] == node && verify_ids[c] == token) {
                node = c;
                break;
            }
        }
    }
}

} // namespace dflash_candidate_stats
} // namespace ninfer::targets::qwen3_6::detail
