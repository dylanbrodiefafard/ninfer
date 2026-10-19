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
#include <stdexcept>
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

inline constexpr int kMaxDrafts = 8;
inline constexpr int kMaxWidth  = 16;
inline constexpr int kVocabCap  = 248320;

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
    std::uint64_t rejects            = 0;
    std::uint64_t reject_in_tree     = 0;
    std::uint64_t reject_top16       = 0;
    std::uint64_t reject_top64       = 0;
    std::uint64_t reject_top256      = 0;
    std::uint64_t reject_in_head     = 0;
    std::uint64_t reject_absent_head = 0;
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
    p.printed      = true;
    const auto pct = [&](std::uint64_t n) {
        return 100.0 * static_cast<double>(n) / static_cast<double>(p.hops);
    };
    const auto rpct = [&](std::uint64_t n) {
        return p.rejects == 0 ? 0.0
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
    int better = 0;
    const std::uint16_t* col =
        p.logits.data() + static_cast<std::size_t>(depth) * static_cast<std::size_t>(p.rows);
    for (int r = 0; r < p.rows; ++r) {
        const float v = bf16_f32(col[r]);
        if (v > target || (v == target && r < row)) { ++better; }
    }
    return better;
}

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
    p.rows              = logits.ne[0];
    p.drafts            = drafts;
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
        p.logged_health    = true;
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
                     p.rows, p.drafts, p.logits.size(), static_cast<unsigned long long>(nans),
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
                if (tok >= 0 && tok < kVocabCap) { p.token_row[static_cast<std::size_t>(tok)] = r; }
            }
        }
    }
}

inline void capture_activation(const char* name, const Tensor& tensor, cudaStream_t stream) {
    if (!dflash_candidate_stats_enabled() || tensor.data == nullptr ||
        tensor.dtype != DType::BF16) {
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
    std::uint64_t nans  = 0;
    std::uint64_t infs  = 0;
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

// Selector lattice dump for offline tree replay. NINFER_DFLASH_SELECTOR_DUMP=<path> (with the
// candidate-stats probe enabled, eager C=1 rounds) appends one record per round:
//   i32 magic 0x44465332, i32 k, i32 anchor, i32 licensed_count, i32 licensed[licensed_count],
//   i32 chain_path[k], then per draft column t: i32 ids[16], f32 unary[16], f32 column
//   log-sum-exp over the represented draft head, f32 pair scores: [16] from the anchor at t=0,
//   [16 parents of column t-1][16 candidates] at t>0.
// pair = sum_r pred[r,parent] * h[r,t] * succ[r,cand] (FP64 host evaluation of the selector's
// bilinear term); the selector's Markov score is unary + pair.
// Then the target verify columns of the same round (row 0): i32 width, and per column j
// (target distribution after the anchor and drafts 1..j): f32 log-sum-exp of the full logits at
// T=1, f32 log-sum-exp at T=2, i32 ids[kTargetTop], f32 logits[kTargetTop] (top logits, any
// order). The p-less cut lies far inside the top kTargetTop at T=2.
inline const char* selector_dump_path() {
    static const char* path = std::getenv("NINFER_DFLASH_SELECTOR_DUMP");
    return path;
}

struct SelectorPending {
    bool valid = false;
    int k      = 0;
    int anchor = 0;
    std::vector<std::int32_t> ids; // [k][16]
    std::vector<float> unary;      // [k][16]
    std::vector<float> lse;        // [k]
    std::vector<float> pair;       // anchor row [16], then [k-1][16][16]
    int target_width = 0;
    std::vector<float> target_lse;        // [width][2]: T=1, T=2
    std::vector<std::int32_t> target_ids; // [width][kTargetTop]
    std::vector<float> target_logits;     // [width][kTargetTop]
};

inline constexpr int kTargetTop = 256;

inline SelectorPending& selector_pending() {
    static SelectorPending s;
    return s;
}

// hidden_proj: BF16 [256,k] device (the selector's h). sel_ids: I32 [16,k] device. pred/succ:
// BF16 [256,248320] device codebooks (rank fastest). anchor: I32 [1] device.
inline void capture_selector(const Tensor& hidden_proj, const Tensor& sel_ids,
                             const Tensor& pred_code, const Tensor& succ_code, const Tensor& anchor,
                             int k, cudaStream_t stream) {
    if (!dflash_candidate_stats_enabled() || selector_dump_path() == nullptr) { return; }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) { return; }
    if (pred_code.data == nullptr || pred_code.dtype != DType::BF16) {
        throw std::logic_error("selector dump requires BF16 selector codebooks");
    }
    constexpr int kRank = 256;
    constexpr int kTop  = 16;
    // The first captured round copies both BF16 selector codebooks ([256, 248320] each, about
    // 254 MB together) to host once, so later rounds score candidate pairs without device reads.
    static std::vector<std::uint16_t> pred_host;
    static std::vector<std::uint16_t> succ_host;
    if (pred_host.empty()) {
        pred_host.resize(static_cast<std::size_t>(pred_code.numel()));
        succ_host.resize(static_cast<std::size_t>(succ_code.numel()));
        CUDA_CHECK(cudaMemcpy(pred_host.data(), pred_code.data, pred_host.size() * 2,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(succ_host.data(), succ_code.data, succ_host.size() * 2,
                              cudaMemcpyDeviceToHost));
    }
    std::vector<std::uint16_t> h(static_cast<std::size_t>(kRank) * static_cast<std::size_t>(k));
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kTop) * static_cast<std::size_t>(k));
    std::int32_t anchor_id = 0;
    CUDA_CHECK(
        cudaMemcpyAsync(h.data(), hidden_proj.data, h.size() * 2, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(ids.data(), sel_ids.data, ids.size() * 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&anchor_id, anchor.data, 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Probe& p = probe();
    std::lock_guard<std::mutex> lock(p.mu);
    SelectorPending& s = selector_pending();
    s.valid            = true;
    s.k                = k;
    s.anchor           = anchor_id;
    s.ids              = ids;
    s.unary.assign(ids.size(), 0.0f);
    s.lse.assign(static_cast<std::size_t>(k), 0.0f);
    s.pair.clear();
    const std::size_t rows = static_cast<std::size_t>(p.rows);
    for (int t = 0; t < k; ++t) {
        const std::uint16_t* col = p.logits.data() + static_cast<std::size_t>(t) * rows;
        float mx                 = -std::numeric_limits<float>::infinity();
        for (std::size_t r = 0; r < rows; ++r) { mx = std::max(mx, bf16_f32(col[r])); }
        double sum = 0.0;
        for (std::size_t r = 0; r < rows; ++r) {
            const float v = bf16_f32(col[r]);
            if (std::isfinite(v)) { sum += std::exp(static_cast<double>(v - mx)); }
        }
        s.lse[static_cast<std::size_t>(t)] = mx + static_cast<float>(std::log(sum));
        for (int c = 0; c < kTop; ++c) {
            const int tok = ids[(static_cast<std::size_t>(t) * kTop + static_cast<std::size_t>(c))];
            const int row =
                tok >= 0 && tok < kVocabCap ? p.token_row[static_cast<std::size_t>(tok)] : -1;
            s.unary[(static_cast<std::size_t>(t) * kTop + static_cast<std::size_t>(c))] =
                row >= 0 ? bf16_f32(col[static_cast<std::size_t>(row)])
                         : -std::numeric_limits<float>::infinity();
        }
    }
    const auto score = [&](int parent, int cand, int t) {
        const std::uint16_t* pr = pred_host.data() + static_cast<std::size_t>(parent) * kRank;
        const std::uint16_t* sr = succ_host.data() + static_cast<std::size_t>(cand) * kRank;
        const std::uint16_t* hr = h.data() + static_cast<std::size_t>(t) * kRank;
        double acc              = 0.0;
        for (int r = 0; r < kRank; ++r) {
            acc += static_cast<double>(bf16_f32(pr[r])) * static_cast<double>(bf16_f32(hr[r])) *
                   static_cast<double>(bf16_f32(sr[r]));
        }
        return static_cast<float>(acc);
    };
    for (int c = 0; c < kTop; ++c) {
        s.pair.push_back(score(anchor_id, ids[static_cast<std::size_t>(c)], 0));
    }
    for (int t = 1; t < k; ++t) {
        for (int a = 0; a < kTop; ++a) {
            const int parent =
                ids[(static_cast<std::size_t>(t - 1) * kTop + static_cast<std::size_t>(a))];
            for (int c = 0; c < kTop; ++c) {
                s.pair.push_back(score(
                    parent, ids[(static_cast<std::size_t>(t) * kTop + static_cast<std::size_t>(c))],
                    t));
            }
        }
    }
}

// target_logits: BF16 [V, width_capacity, B] device; captures row 0's first `width` columns.
inline void capture_target_logits(const Tensor& target_logits, int width) {
    if (!dflash_candidate_stats_enabled() || selector_dump_path() == nullptr) { return; }
    SelectorPending& s = selector_pending();
    if (!s.valid) { return; }
    const std::size_t vocab = static_cast<std::size_t>(target_logits.ne[0]);
    std::vector<std::uint16_t> bits(vocab * static_cast<std::size_t>(width));
    CUDA_CHECK(
        cudaMemcpy(bits.data(), target_logits.data, bits.size() * 2, cudaMemcpyDeviceToHost));
    s.target_width = width;
    s.target_lse.assign(static_cast<std::size_t>(width) * 2, 0.0f);
    s.target_ids.assign(static_cast<std::size_t>(width) * kTargetTop, 0);
    s.target_logits.assign(static_cast<std::size_t>(width) * kTargetTop, 0.0f);
    std::vector<float> col(vocab);
    std::vector<std::int32_t> order(vocab);
    for (int j = 0; j < width; ++j) {
        float mx = -std::numeric_limits<float>::infinity();
        for (std::size_t v = 0; v < vocab; ++v) {
            col[v] = bf16_f32(bits[static_cast<std::size_t>(j) * vocab + v]);
            if (std::isfinite(col[v])) { mx = std::max(mx, col[v]); }
        }
        double s1 = 0.0;
        double s2 = 0.0;
        for (std::size_t v = 0; v < vocab; ++v) {
            if (!std::isfinite(col[v])) { continue; }
            s1 += std::exp(static_cast<double>(col[v] - mx));
            s2 += std::exp(static_cast<double>(col[v] - mx) / 2.0);
        }
        s.target_lse[static_cast<std::size_t>(j) * 2] = mx + static_cast<float>(std::log(s1));
        s.target_lse[static_cast<std::size_t>(j) * 2 + 1] =
            mx / 2.0f + static_cast<float>(std::log(s2));
        for (std::size_t v = 0; v < vocab; ++v) { order[v] = static_cast<std::int32_t>(v); }
        std::nth_element(order.begin(), order.begin() + kTargetTop, order.end(),
                         [&](std::int32_t a, std::int32_t b) {
                             const float x = std::isfinite(col[static_cast<std::size_t>(a)])
                                                 ? col[static_cast<std::size_t>(a)]
                                                 : -1e30f;
                             const float y = std::isfinite(col[static_cast<std::size_t>(b)])
                                                 ? col[static_cast<std::size_t>(b)]
                                                 : -1e30f;
                             return x > y;
                         });
        for (int i = 0; i < kTargetTop; ++i) {
            const std::int32_t v = order[static_cast<std::size_t>(i)];
            s.target_ids[static_cast<std::size_t>(j) * kTargetTop + static_cast<std::size_t>(i)] =
                v;
            s.target_logits[static_cast<std::size_t>(j) * kTargetTop +
                            static_cast<std::size_t>(i)] = col[static_cast<std::size_t>(v)];
        }
    }
}

inline void write_selector_record(const std::int32_t* verify_ids, const std::int32_t* licensed,
                                  int licensed_count) {
    SelectorPending& s = selector_pending();
    if (!s.valid || selector_dump_path() == nullptr) { return; }
    s.valid             = false;
    static std::FILE* f = [] {
        std::FILE* file = std::fopen(selector_dump_path(), "ab");
        if (file == nullptr) { throw std::runtime_error("cannot open selector dump"); }
        return file;
    }();
    const std::int32_t header[4] = {0x44465332, s.k, s.anchor, licensed_count};
    std::fwrite(header, 4, 4, f);
    std::fwrite(licensed, 4, static_cast<std::size_t>(licensed_count), f);
    std::fwrite(verify_ids + 1, 4, static_cast<std::size_t>(s.k), f);
    for (int t = 0; t < s.k; ++t) {
        std::fwrite(s.ids.data() + static_cast<std::size_t>(t) * 16, 4, 16, f);
        std::fwrite(s.unary.data() + static_cast<std::size_t>(t) * 16, 4, 16, f);
        std::fwrite(s.lse.data() + t, 4, 1, f);
        const std::size_t off = t == 0 ? 0 : 16 + static_cast<std::size_t>(t - 1) * 256;
        std::fwrite(s.pair.data() + off, 4, t == 0 ? 16 : 256, f);
    }
    std::fwrite(&s.target_width, 4, 1, f);
    for (int j = 0; j < s.target_width; ++j) {
        std::fwrite(s.target_lse.data() + static_cast<std::size_t>(j) * 2, 4, 2, f);
        std::fwrite(s.target_ids.data() + static_cast<std::size_t>(j) * kTargetTop, 4, kTargetTop,
                    f);
        std::fwrite(s.target_logits.data() + static_cast<std::size_t>(j) * kTargetTop, 4,
                    kTargetTop, f);
    }
    s.target_width = 0;
    std::fflush(f);
}

inline void record_round(const std::int32_t* verify_ids, const std::int32_t* parents,
                         const std::int32_t* licensed, int licensed_count, int width, int drafts) {
    if (!dflash_candidate_stats_enabled() || licensed_count <= 0) { return; }
    Probe& p = probe();
    std::lock_guard<std::mutex> lock(p.mu);
    if (p.logits.empty() || p.rows <= 0) { return; }
    write_selector_record(verify_ids, licensed, licensed_count);
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
