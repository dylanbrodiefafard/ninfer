#pragma once

#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include <ninfer/targets/qwen3_6/frontend.h>

#include <cuda_runtime_api.h>
#include <exception>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

// A constraint failure at node safe_columns licenses only preceding columns. The
// exception belongs to that node, not to the batch; a zero prefix is a root error.
struct TokenMaskResult {
    std::size_t safe_columns = 0;
    std::exception_ptr error;
};

// CPU-only mask fill. RequestErrors are attributed to the first failing node by
// retrying prefixes only on the error path. Failed columns receive all-ones masks
// so device work can finish, but callers must not publish those columns. Other
// exceptions denote broken invariants and propagate. Inputs use parent-before-child
// topology and words holds exactly tokens.size() vocabulary bitsets.
[[nodiscard]] TokenMaskResult fill_token_mask_row(const OutputSession& output,
                                                  const ops::SamplingConfig& config,
                                                  std::span<const TokenId> tokens,
                                                  std::span<const std::int32_t> parents,
                                                  std::span<std::uint32_t> words);

// Program-owned, startup-sized exchange. Its address and pinned/device buffers
// remain stable for every captured graph. Host copies are fixed-size 1D copies of
// device staging: drivers may refuse to update 2D graph copies that touch host
// memory. The executor changes bindings only at synchronized round boundaries;
// the CUDA host node makes no CUDA API calls. Program teardown and error paths
// drain the submission streams before destroying bound OutputSessions. This shared
// frontend transport owns no model execution or speculative acceptance policy.
class TokenMaskExchange {
public:
    // Initializes staging on `compute`; submissions use that stream or wait for
    // its startup work before using the exchange on another compute stream.
    TokenMaskExchange(Tensor masks, Tensor sampling, Tensor nodes, cudaStream_t compute);
    ~TokenMaskExchange();
    TokenMaskExchange(const TokenMaskExchange&)            = delete;
    TokenMaskExchange& operator=(const TokenMaskExchange&) = delete;
    // Session policies replace eligibility in the supplied configs. During enqueue,
    // rows without a session retain their supplied domain; a one-shot bitset requires
    // a single valid column. Bindings and sessions stay immutable until streams drain.
    void bind(std::span<const OutputSession* const> outputs,
              std::span<const ops::SamplingConfig> sampling);
    // Ordinary/prefill root sampling; called at a synchronized CPU boundary.
    [[nodiscard]] ops::SamplingConfig root(std::size_t row, cudaStream_t stream);

    // Called inside the speculative graph after ids/parents are constructed and
    // before target compute. The exchange forks onto `host`; the caller joins the
    // returned ready event before target sampling. The returned configs are
    // target-only: draft proposal sampling keeps its own unmasked configs.
    struct Submission {
        const ops::SamplingConfig* sampling;
        cudaEvent_t ready;
    };

    [[nodiscard]] Submission enqueue(const Tensor& ids, const Tensor* parents,
                                     const Tensor& valid_columns, cudaStream_t compute,
                                     cudaStream_t host);
    // After synchronizing: transport/invariant errors throw; constraint errors are
    // returned per row with their safe prefix. Sessions remain unchanged.
    [[nodiscard]] std::span<const TokenMaskResult> results() const;
    // Consumers without row-local resolution fail before publishing any candidate.
    void rethrow_error() const;

private:
    static void CUDART_CB match(void* opaque) noexcept;
    void fill();
    [[nodiscard]] std::uint32_t* host_mask(std::size_t row) const;
    [[nodiscard]] const std::uint32_t* device_mask(std::size_t row) const;

    cudaEvent_t candidates_ready_ = nullptr;
    cudaEvent_t masks_ready_      = nullptr;
    Tensor masks_;
    Tensor sampling_;
    Tensor nodes_;
    const std::size_t width_;
    const std::size_t capacity_;
    PinnedHostBuffer host_masks_;
    PinnedHostBuffer host_sampling_;
    PinnedHostBuffer host_ids_;
    PinnedHostBuffer host_parents_;
    PinnedHostBuffer host_counts_;
    std::vector<const OutputSession*> outputs_;
    std::vector<ops::SamplingConfig> configs_;
    std::vector<TokenMaskResult> results_;
    std::exception_ptr error_;
    bool tree_ = false;
};

} // namespace ninfer::targets::qwen3_6
