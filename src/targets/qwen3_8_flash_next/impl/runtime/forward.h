#pragma once

// The Qwen3.8-Flash-Next forward pass of 1..12 lanes: eager prefill super-chunks of one lane and
// lane-batched decode rounds (one CUDA Graph per batch size) over paged QSA state, with optional
// noncommitting chain recording for speculative qualification.
// The target package owns this Program; see docs/maintainer/qwen3.8-flash-next-plan.md, "Execution
// design".

#include "model_view.h"
#include "runtime/cuda_event.h"
#include "runtime/expert_residency.h"
#include "runtime/ple_feeder.h"
#include "runtime/state.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_expert_cache.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h> // cudaStream_t, cudaEvent_t

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

class Dflash;

// Whether an internal prefill boundary needs a tail distribution for exact continuation.
enum class PrefillTail : std::uint8_t { Skip, Capture };

/** Which token columns of a chunk produce vocabulary logits. */
enum class LogitColumns : std::uint8_t {
    // Only the last token of a prefill (its final chunk); the decode case.
    Last,
    // Every token of every chunk; teacher-forced evaluation.
    All,
};

struct ForwardOptions {
    // Widest sub-chunk, in [1, ops::kMoeMaxTokens]: the token count each layer's Ops take at once.
    // Per-call activations and workspaces are sized for it.
    std::int32_t max_chunk_tokens = 4096;
    // Widest super-chunk, at least max_chunk_tokens: a prefill runs layer-major over this many
    // tokens (each layer over all of the super-chunk's sub-chunks, then the next layer), so a
    // layer's streamed expert records serve every sub-chunk. Its token inputs, n-gram rows and
    // four-stream residual (20 KiB per token) are held for the whole super-chunk.
    std::int32_t super_chunk_tokens = 16384;
    // Complete four-token blocks a QSA query may select beyond its tail.
    ops::QsaSelectBudget select_budget = ops::QsaSelectBudget::Blocks512;
    LogitColumns logits                = LogitColumns::Last;
    // Startup-fixed chain-record capacity; 0 disables recording, otherwise 2..8 columns per lane.
    std::int32_t max_verify_width   = 0;
    DflashExecution dflash_features = DflashExecution::Disabled;
    // Diagnostic capture of packed verification routes, for expert-admission policy replay.
    bool trace_verify_routes = false;
    // Super-chunks of at least window_min_tokens stream each layer's non-resident expert records
    // through one of two VRAM windows of expert_window_bytes (a multiple of 256) and admit
    // nothing; narrower ones take the decode path (admit, fill, commit). The windows are F3 cache
    // slots borrowed for each such super-chunk; they stream only when expert_window_bytes > 0,
    // super_chunk_tokens >= window_min_tokens >= 2 and the F3 pool holds both windows. The
    // defaults (sub-chunk 4096, super-chunk 16384, window from 512 tokens) are the measured choice
    // in docs/maintainer/qwen3.8-flash-next-plan.md ("Prefill measurements").
    std::size_t expert_window_bytes = 0;
    std::int32_t window_min_tokens  = 512;
};

/**
 * Called after a head with BF16 logits [248320, n] on the device (n = 1 for LogitColumns::Last,
 * once per prefill; else the sub-chunk width, once per sub-chunk) and the position of column 0. The
 * tensor is valid until the next forward call; its values are complete once the forward stream
 * reaches this point, so the callback synchronizes the stream before reading it.
 */
using LogitsSink = std::function<void(const Tensor& logits, std::int32_t first_position)>;

/**
 * Optional diagnostic taps of eager chunks (prefill), under the same validity and stream rules as
 * LogitsSink; an empty function is not called.
 *
 *   after_route  after each layer's moe_route, with the routed expert ids I32 [10,T] (k, token);
 *   after_mixer  before mixer injection, with BF16 input/output [2560,T] and the first position;
 *   after_layer  after every layer, with that layer's BF16 residual [2560,4,T] (stream-major
 *                within a token) and the chunk's first position.
 */
struct ForwardObserver {
    std::function<void(std::int32_t layer, const Tensor& expert_ids)> after_route;
    std::function<void(std::int32_t layer, const Tensor& input, const Tensor& output,
                       std::int32_t first_position)>
        after_mixer;
    std::function<void(std::int32_t layer, const Tensor& residual, std::int32_t first_position)>
        after_layer;
};

/**
 * Expert-cache events since the last clear_expert_cache_counts, indexed by ops::MoeExpertCodec:
 * per admitting layer call (streamed super-chunks admit nothing), each distinct routed expert is
 * one lookup, and a miss when its record is not
 * resident, so its moe_expert_record_bytes cross PCIe once (moe_expert_cache_admit's counters).
 */
struct ExpertCacheCounts {
    std::array<std::int64_t, ops::kMoeCacheCodecTags> lookups{};
    std::array<std::int64_t, ops::kMoeCacheCodecTags> misses{};
};

// The pool whose slots the prefill streaming windows borrow: the F3 pool, the larger one, whose
// 256-aligned slots hold records of either codec packed by moe_expert_cache_window.
inline constexpr ops::MoeExpertCodec kWindowCodec = ops::MoeExpertCodec::F3G32;

// Ranked alternatives each sampled token reports (ops::token_logprobs' K).
inline constexpr std::int32_t kReportedAlternatives = 20;

/** How decode runs a round: by replaying the captured graph of its batch size, or eagerly. */
enum class DecodeExecution : std::uint8_t { Graph, Eager };

/** Whether a decode round evaluates its rows' log probabilities and ranked alternatives. */
enum class RoundLogprobs : std::uint8_t { Report, Skip };

/**
 * One sampled token per row of a sampling call, valid until the next prefill_step, sample_last or
 * decode: the token, its log probability and the kReportedAlternatives most probable ids with
 * their log probabilities under the unmodified distribution (temperature 1, every token of the
 * domain, no penalty or eligibility mask), descending with the lower id first on ties.
 */
struct SampledRows {
    std::span<const std::int32_t> tokens;
    std::span<const float> logprobs;
    std::span<const std::int32_t> top_ids; // kReportedAlternatives per row, row-major
    std::span<const float> top_logprobs;   // kReportedAlternatives per row, row-major
};

/**
 * Runs the lanes' tokens through the 48 layers and the head on one stream, with the expert-cache
 * copies forked onto a second one and joined before the moe_experts that reads them, so every
 * call returns with the second stream joined.
 *
 * Prefill appends one lane's tokens in super-chunks of at most super_chunk_tokens tokens, each cut
 * into sub-chunks of at most max_chunk_tokens. Per super-chunk at positions [p, p + T): the
 * embedding and hyper_connection_broadcast of every sub-chunk into the super-chunk's four-stream
 * residual, then layer by layer, for each sub-chunk in order (a layer's GDN and QSA state
 * advance sub-chunk by sub-chunk, and a sub-chunk's QSA selection sees the prefix through its
 * own end): (layer 1 only) the n-gram embedding's ple_inject; the mixer read, GDN or QSA mixer
 * and inject; the MoE read, routing, moe_expert_cache_admit, moe_expert_cache_fill of the admitted
 * misses' records into their slots on a second stream, moe_experts (resident records from their
 * slots or, when admit could not place them, from mapped host memory while the fill runs, then
 * the admitted misses from their slots once it completes), moe_expert_cache_commit, and inject.
 * The head is the final read-only hyper-connection read and the vocabulary projection.
 *
 * A super-chunk of at least window_min_tokens tokens (with windows configured) routes to nearly
 * every expert of each layer, so it admits nothing: layer l's non-resident records are copied once
 * into window l % 2 on the fill stream while earlier layers compute (moe_expert_cache_window
 * places them, moe_expert_cache_fill copies them; the copy of layer l + 2 starts once layer l's
 * last sub-chunk has read the window), and every sub-chunk's moe_experts reads them through its
 * deferral. Records past the window stay on mapped reads. The windows' bytes are two runs of
 * consecutive F3 cache slots, borrowed for the super-chunk from just ahead of the CLOCK hand
 * (moe_expert_cache_borrow evicts them, and no admit takes them while lent) and returned empty
 * after its last layer, so between windowed super-chunks, and in decode rounds of other lanes
 * between two of them, the whole pool serves the cache; the rest of the cache is left as the last
 * admitting call left it.
 *
 * A decode round runs B open lanes' next tokens as B rows of one column through the same layer
 * sequence: GDN through the slot-indexed snapshot forms in place (each lane's slot), QSA over the
 * lanes' block-table rows, ple_inject over the lanes' history slots, and one routed-expert call
 * over all B tokens, whose admit sees the union of their experts. Sampling (ops::sample with each
 * row's configuration) and ops::token_logprobs follow the head. With LogitColumns::Last, one graph
 * per batch size 1 .. state.lanes() is captured at construction; each uploads the round's ingress
 * frame (ids, positions, lanes, sampling configurations) and n-gram rows from fixed pinned buffers
 * and copies the egress back. QSA selection is captured for the lane capacity: the uploaded slot
 * sets each replay's visible prefix.
 *
 * The model, state, feeder and expert residency must outlive the Forward, which owns the chunk
 * activations, the transient workspace (sized at construction for max_chunk_tokens, the lane
 * count and the lane capacity), and the pinned staging. Calls come from one thread. Errors throw
 * before the offending chunk is enqueued where they are argument errors; a CUDA failure throws
 * and leaves the involved lanes' state unspecified (they must be closed).
 */
struct SpeculativeRows {
    std::int32_t width = 0;
    std::span<const std::int32_t> tokens;
    std::span<const std::int32_t> counts;
    std::span<const std::int32_t> accepted;
    std::span<const float> calibration;
    std::span<const float> logprobs;
    std::span<const std::int32_t> top_ids;
    std::span<const float> top_logprobs;
};

class Forward {
public:
    Forward(const ModelView& model, State& state, PleRowFeeder& feeder, ExpertResidency& experts,
            const ForwardOptions& options, cudaStream_t stream, Dflash* drafter = nullptr);

    // Throws std::invalid_argument unless the options are valid (the constructor's check).
    static void validate(const ForwardOptions& options);

    // Device bytes a Forward with these options allocates over `state` (its lane count and lane
    // capacity): the super-chunk and sub-chunk activations, the decode frames and the transient
    // workspace (the streaming windows borrow cache slots). CUDA Graph instantiation is not
    // included.
    [[nodiscard]] static std::size_t
    device_bytes(const ModelView& model, const ForwardOptions& options, const State& state);

    // Waits for the stream and the fill stream before releasing the buffers they use.
    ~Forward();

    Forward(const Forward&)            = delete;
    Forward& operator=(const Forward&) = delete;

    /**
     * Appends the first n = min(remaining.size(), super_chunk_tokens) tokens of `remaining` (each
     * id in [0, 248320), committed + n <= the lane's reservation) to the open `lane` as one
     * super-chunk and returns n. The super-chunk's n-gram rows come from the feeder's prefetch
     * when it holds exactly them for this lane, else from a gather started here (any other pending
     * gather is cancelled); once the last part is uploaded, the gather of the following
     * super-chunk of `remaining` begins on the feeder's thread. Logits go to `sink` (when not
     * empty) per the configured LogitColumns: All, once per sub-chunk in order; Last, only when n
     * == remaining.size(), for the last token, which also leaves that token's logits for
     * sample_last. A positive token_limit bounds an internal checkpoint segment without dropping
     * the remaining lookahead; PrefillTail::Capture also computes and saves its final logits.
     * Synchronizes the stream before returning, so the step's device work is complete
     * (the following super-chunk's row gather may still run on the feeder's thread).
     */
    std::int32_t prefill_step(std::int32_t lane, std::span<const std::int32_t> remaining,
                              const LogitsSink& sink, const ForwardObserver* observer = nullptr,
                              std::int32_t token_limit = 0, PrefillTail tail = PrefillTail::Skip);

    // prefill_step until `tokens` (non-empty) are all appended.
    void prefill(std::int32_t lane, std::span<const std::int32_t> tokens, const LogitsSink& sink,
                 const ForwardObserver* observer = nullptr);

    /**
     * Samples the next token after a prefill_step that completed a prompt with LogitColumns::Last:
     * ops::sample over [0, kTokenDomain) of the last token's logits with `config` (device pointers
     * as ops::SamplingConfig requires) at `position`, the last prompt token's position (the RNG
     * key, kSamplePurposePrefill), then ops::token_logprobs. Synchronizes the stream; the result
     * has one row.
     */
    [[nodiscard]] SampledRows sample_last(const ops::SamplingConfig& config, std::int32_t position);
    // Finalizes an exact-prefix hit from the lane's retained BF16 vocabulary logits, without
    // changing its model state. Sampling configuration and RNG belong to the incoming request.
    [[nodiscard]] SampledRows sample_retained(std::int32_t lane, const ops::SamplingConfig& config);

    /**
     * One decode round: row i appends tokens[i] (in [0, 248320)) to open lane lanes[i] (distinct,
     * committed < the lane's reservation) at its committed position, then samples its next token
     * with configs[i] (kSamplePurposeDecode, keyed by that position) and advances the lane by one.
     * With RoundLogprobs::Skip, token_logprobs skips every row and the result's log
     * probabilities and alternatives are unspecified.
     * The host gathers the rows' n-gram records into the pinned row frame, writes the ingress
     * frame, and replays the graph of batch size B = lanes.size() (DecodeExecution::Graph,
     * LogitColumns::Last only) or enqueues the same work eagerly; the call then synchronizes. The
     * round's logits stay in logits() until the next call. An observer (DecodeExecution::Eager
     * only; a graph replay cannot call the host) sees each layer's routes I32 [10,B] and residual
     * with the ForwardObserver rules. Requires 1 <= B <= state.lanes() and no
     * pending feeder gather of a decoding lane.
     */
    [[nodiscard]] SampledRows decode(std::span<const std::int32_t> lanes,
                                     std::span<const std::int32_t> tokens,
                                     std::span<const ops::SamplingConfig> configs,
                                     RoundLogprobs logprobs          = RoundLogprobs::Report,
                                     DecodeExecution execution       = DecodeExecution::Graph,
                                     const ForwardObserver* observer = nullptr);

    /**
     * Evaluates packed chain inputs [width, lanes.size()] without committing their GDN, PLE,
     * n-gram histories or logical frontiers. Requires distinct open lanes, width in
     * [1,max_verify_width], and one valid prefix length in [1,width] per row, fitting its reserved
     * QSA capacity. Invalid-tail tokens must still be valid vocabulary IDs. Provisional QSA slots
     * beyond the frontier are overwritten by subsequent execution. The returned vocabulary logits
     * cover every physical column and are valid until the next forward call; invalid tails are
     * unspecified. Synchronizes the stream. Recording buffers belong to this Forward.
     */
    [[nodiscard]] Tensor verify(std::span<const std::int32_t> lanes,
                                std::span<const std::int32_t> tokens, std::int32_t width,
                                std::span<const std::int32_t> valid_columns);

    // Commits each row's prefix of the most recent verify() call, before any other forward.
    // Counts in [0,valid_columns[b]] are authoritative Engine publication counts. Zero is a
    // strict no-op on canonical model and raw-token state. Calls synchronize before returning.
    void commit_verify(std::span<const std::int32_t> commit_counts);
    // Most recent call's feature columns, in chronological/request-major order. Prefill retains
    // only its final 2048 columns; decode and verification retain every physical column.
    [[nodiscard]] Tensor dflash_features() const;
    [[nodiscard]] SpeculativeRows speculate(std::span<const std::int32_t> lanes,
                                            std::span<const std::int32_t> anchors,
                                            std::span<const ops::SamplingConfig> configs,
                                            std::int32_t k, std::span<const std::int32_t> extents,
                                            RoundLogprobs logprobs);
    void resolve_speculative(std::span<const std::int32_t> counts);
    // I32 [10,width*rows] routed ids for one layer of the latest unresolved verify frame.
    // Storage remains owned by Forward until the next forward. Throws std::logic_error unless
    // trace_verify_routes is enabled, the frame exists and layer is in [0,kLayers).
    [[nodiscard]] Tensor verify_routes(std::int32_t layer) const;

    // Host seconds of the last decode round's n-gram gather (including the wait for a pending
    // prefetch's reads) and the GPU milliseconds of its graph replay or eager work.
    [[nodiscard]] double last_gather_seconds() const noexcept { return last_gather_seconds_; }

    [[nodiscard]] float last_replay_ms() const noexcept { return last_replay_ms_; }

    /**
     * Teacher-forced log p(tokens[i] | tokens[0..i)) for i in [1, tokens.size()), tokens.size() >=
     * 2, on the open `lane`: prefills tokens[0 .. size - 1) and evaluates the head over every
     * position in groups of the logits width, with ops::token_logprobs against the next token.
     * Synchronizes the stream.
     */
    [[nodiscard]] std::vector<float> score(std::int32_t lane, std::span<const std::int32_t> tokens);

    // The head's last BF16 logits [248320, columns] (decode: one column per row of the last
    // round); valid until the next call.
    [[nodiscard]] Tensor logits(std::int32_t count) const;

    // Zeroes the expert-cache counters, which every admit (decode rounds and super-chunks narrower
    // than window_min_tokens) accumulates from construction on; enqueued on the stream.
    void clear_expert_cache_counts();

    // The counters once the stream's enqueued work completes; synchronizes the stream.
    [[nodiscard]] ExpertCacheCounts expert_cache_counts();

    [[nodiscard]] std::size_t activation_bytes() const noexcept { return activations_.capacity(); }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept { return work_.capacity(); }

    // Bytes the two streaming windows borrow from the expert cache while a windowed super-chunk
    // runs (0 when none stream).
    [[nodiscard]] std::size_t window_bytes() const noexcept;

private:
    // One sub-chunk's host-supplied inputs. The blocks of a super-chunk are allocated first, in
    // order, from the activation arena's base, so the pinned staging mirrors [base, end of the
    // last block) and the first n blocks upload in one copy.
    struct ChunkInputs {
        Tensor token_ids; // I32 [max]
        Tensor slots;     // I32 [max], the QSA state slot of each token (its position)
        Tensor
            positions; // I32 [max,3] storage; a sub-chunk of T views [T,3] axis-major at its start
    };

    /**
     * The decode round's host-written frame, uploaded whole (pinned mirror to a device copy) at
     * the start of every round; row i of a batch of B uses entry i of each array:
     *   tokens          the row's input token id;
     *   lanes           its lane (QSA block-table row, GDN, PLE and count slot);
     *   slots           its position: QSA slot, rotary position and sampling RNG position;
     *   qsa_positions   the QSA Ops' [1,3,B] positions (element 3 i + a, axis a);
     *   rope_positions  ops::rope's [B,3] axis-major positions (element a B + i), written for the
     *                   round's B;
     *   sampling        the row's ops::SamplingConfig;
     *   logprobs        1 when token_logprobs evaluates the row (its row_enabled), else 0.
     * Every array starts 16-byte aligned, as the Ops require.
     */
    struct DecodeIngress {
        alignas(16) std::array<std::int32_t, kMaxLanes> tokens;
        alignas(16) std::array<std::int32_t, kMaxLanes> lanes;
        alignas(16) std::array<std::int32_t, kMaxLanes> slots;
        alignas(16) std::array<std::int32_t, 3 * kMaxLanes> qsa_positions;
        alignas(16) std::array<std::int32_t, 3 * kMaxLanes> rope_positions;
        alignas(16) std::array<ops::SamplingConfig, kMaxLanes> sampling;
        alignas(16) std::array<std::int32_t, kMaxLanes> logprobs;
    };

    // The sampling results, copied back whole after every sampling call (row i at entry i).
    struct SamplingEgress {
        alignas(16) std::array<std::int32_t, kMaxLanes> tokens;
        alignas(16) std::array<float, kMaxLanes> logprobs;
        alignas(16) std::array<std::int32_t, kReportedAlternatives * kMaxLanes> top_ids;
        alignas(16) std::array<float, kReportedAlternatives * kMaxLanes> top_logprobs;
    };

    // A batch of B rows of W columns through the layers: a prefill sub-chunk (B = 1, W tokens of
    // one lane) or a decode round (W = 1, B lanes). Column j of row b is packed column t = b W + j
    // of every [.., T] tensor, T = W B.
    enum class StateMode : std::uint8_t { Commit, Record };

    struct SubChunk {
        // Input block and feeder part of a prefill sub-chunk (0 for decode).
        std::int32_t index = 0;
        // First column of the batch in the super-chunk tensors (0 for decode).
        std::int32_t offset = 0;
        std::int32_t width  = 0;
        std::int32_t rows   = 0;
        // Position of column 0 (prefill observers) and the bound of every query's visible prefix.
        std::int32_t first_position = 0;
        std::int32_t max_visible    = 0;
        // The prefill lane (host), or -1 for a decode round.
        std::int32_t lane = -1;
        Tensor token_ids;      // I32 [T]
        Tensor rope_positions; // I32 [T,3] axis-major
        Tensor slots;          // I32 [W,B]
        Tensor qsa_positions;  // I32 [W,3,B]
        Tensor lanes;          // I32 [B]
        StateMode state_mode = StateMode::Commit;
        Tensor valid_columns;      // I32 [B], recording only.
        Tensor ple_snapshot_slots; // I32 [B], recording only.

        [[nodiscard]] std::int32_t tokens() const noexcept { return width * rows; }
    };

    // Persistent tensors, allocated once: the super-chunk ones at super_chunk_tokens columns
    // (S), the sub-chunk ones at max_chunk_tokens columns (max); a batch of T tokens uses the
    // first T columns of each sub-chunk tensor (column-major prefixes are contiguous).
    struct ChunkStorage {
        std::vector<ChunkInputs> input_blocks; // ceil(S / max) blocks
        Tensor ple_rows;                       // U8 [162,16,S]
        Tensor residual;                       // BF16 [2560,4,S]
        Tensor block_input;                    // BF16 [2560,max], the hyper-connection read
        Tensor write_scale;                    // FP32 [4,max]
        Tensor block_output;                   // BF16 [2560,max], mixer or MoE output
        Tensor head_input;                     // BF16 [2560,n]
        Tensor logits;                         // BF16 [248320,n]; n = logit_columns(options)
        // Decode and recording GDN q, k (BF16 [2048,n]), v and z (BF16 [6144,n]), outside the
        // workspace arena as gdn_input_proj_conv_snapshot requires; n covers every decode row
        // and, when enabled, every packed verify column.
        Tensor decode_gdn_query;
        Tensor decode_gdn_key;
        Tensor decode_gdn_value;
        Tensor decode_gdn_z;
        Tensor ingress; // U8 [sizeof(DecodeIngress)]
        // U8 [sizeof(SamplingEgress)]: ops::sample's tokens, then token_logprobs' log
        // probabilities and alternatives, at the egress struct's offsets, copied back whole.
        Tensor egress;
        Tensor row_enabled; // I32 [1] holding 1: score()'s token_logprobs row selector
        // score(): I32 [S] next-token targets, FP32 [S] their log probabilities, and the
        // alternatives token_logprobs reports (one per logits column, unused).
        Tensor score_targets;
        Tensor score_logprobs;
        Tensor score_top_ids;
        Tensor score_top_logprobs;
        // I64 [2,2] moe_expert_cache_admit counters (kind, codec), accumulated by every admit;
        // the decode graphs capture their address.
        Tensor expert_cache_counters;
        // I64 [512,2]: column b holds the placements of the layer streamed through window b.
        Tensor window_columns;
        Tensor dflash_features;
    };

    static constexpr std::int32_t kVerifyMaxColumns = 8 * kMaxLanes;

    struct VerifyIngress {
        alignas(16) std::array<std::int32_t, kVerifyMaxColumns> tokens;
        alignas(16) std::array<std::int32_t, kVerifyMaxColumns> slots;
        alignas(16) std::array<std::int32_t, 3 * kVerifyMaxColumns> rope_positions;
        alignas(16) std::array<std::int32_t, 3 * kVerifyMaxColumns> qsa_positions;
        alignas(16) std::array<std::int32_t, kMaxLanes> lanes;
        alignas(16) std::array<std::int32_t, kMaxLanes> valid_columns;
        alignas(16) std::array<std::int32_t, kMaxLanes> ple_snapshot_slots;
    };

    struct VerifyCommitIngress {
        alignas(16) std::array<std::int32_t, kMaxLanes> selectors;
        alignas(16) std::array<std::int32_t, kMaxLanes> enabled;
        alignas(16) std::array<std::int32_t, kMaxLanes> counts;
    };

    struct SpecIngress {
        alignas(16) std::array<std::int32_t, kMaxLanes> anchors, frontiers, lanes, valid, extents;
        alignas(16) std::array<std::int32_t, kMaxLanes> append_counts, logprobs;
        alignas(16) std::array<std::int32_t, kDflashWidth * kMaxLanes> append_positions;
        alignas(16) std::array<std::int32_t, 8 * kDflashWidth * kMaxLanes> backup_slots;
        alignas(16) std::array<ops::SamplingConfig, kMaxLanes> configs;
    };

    struct SpecEgress {
        alignas(16) std::array<std::int32_t, kDflashMaxK * kMaxLanes> drafts;
        alignas(16) std::array<std::int32_t, kDflashWidth * kMaxLanes> tokens, verify_ids;
        alignas(16) std::array<std::int32_t, kMaxLanes> counts, accepted;
        alignas(16) std::array<float, ops::kPLessProposalCalibrationTemperatureCount * kDflashMaxK *
                                          kMaxLanes> calibration;
        alignas(16) std::array<float, kDflashWidth * kMaxLanes> logprobs;
        alignas(
            16) std::array<std::int32_t, kReportedAlternatives * kDflashWidth * kMaxLanes> top_ids;
        alignas(
            16) std::array<float, kReportedAlternatives * kDflashWidth * kMaxLanes> top_logprobs;
        alignas(16) std::array<std::int32_t, kDflashWidth * kMaxLanes> count_patch;
    };

    struct VerifyStorage {
        Tensor ingress;
        Tensor ple_states; // BF16 [9,10240,C + W*C]; canonical copies precede compact snapshots.
        Tensor convolution_records;
        Tensor key_records;
        Tensor value_records;
        Tensor gate_records;
        Tensor commit_ingress;
        Tensor selected_ple;
        Tensor selected_logits;
        Tensor spec_ingress, spec_egress, targets, selector_ids, selector_q, lengths, anchors;
        Tensor count_snapshot;
        Tensor context_key_backup, context_value_backup;
        Tensor expert_routes;
    };
    template <class Allocator>
    static VerifyStorage allocate_verify(Allocator& allocator, const ForwardOptions& options,
                                         std::int32_t lanes);
    [[nodiscard]] static std::size_t plan_verify_bytes(const ForwardOptions& options,
                                                       std::int32_t lanes);
    [[nodiscard]] SubChunk verify_batch(std::int32_t width, std::int32_t rows) const;
    [[nodiscard]] GdnReplayRecords verify_records(std::int32_t width) const;
    void prepare_verify(std::span<const std::int32_t> lanes, std::span<const std::int32_t> tokens,
                        std::int32_t width, std::span<const std::int32_t> valid_columns);
    void enqueue_spec_a(std::int32_t width, std::int32_t rows);
    void enqueue_spec_b(std::int32_t width, std::int32_t rows);
    void capture_spec_graphs();
    void require_resolved() const;

    // Run against a WorkspaceLayoutBuilder (planning) and the DeviceArena (binding) so both share
    // every size, alignment and order.
    template <class Allocator>
    static std::vector<ChunkInputs> allocate_inputs(Allocator& allocator,
                                                    const ForwardOptions& options);
    template <class Allocator>
    static ChunkStorage allocate_chunk(Allocator& allocator, const ForwardOptions& options);
    [[nodiscard]] static std::size_t plan_input_bytes(const ForwardOptions& options);
    [[nodiscard]] static std::size_t plan_activation_bytes(const ForwardOptions& options);

    // One super-chunk of `tokens` on `lane`; begins the gather of next_tokens (when not empty)
    // after the last part's upload. With score_targets (one per token), evaluates every token's
    // log probability of its target into score_logprobs instead of the configured logits.
    void run_super_chunk(std::int32_t lane, std::span<const std::int32_t> tokens,
                         std::span<const std::int32_t> next_tokens, bool final_super_chunk,
                         const LogitsSink& sink, const ForwardObserver* observer,
                         std::span<const std::int32_t> score_targets);
    // The sub-chunks of a super-chunk of `tokens` tokens of `lane` at first_position.
    [[nodiscard]] std::vector<SubChunk> sub_chunks(std::int32_t lane, std::int32_t tokens,
                                                   std::int32_t first_position) const;
    // The decode round of B rows over the device ingress frame.
    [[nodiscard]] SubChunk decode_batch(std::int32_t rows) const;
    // Writes the ids, slots and positions of `tokens` into the pinned staging, each sub-chunk into
    // its input block.
    void write_inputs(std::span<const SubChunk> chunks, std::span<const std::int32_t> tokens);
    // Uploads input blocks [0, blocks).
    void upload_inputs(std::size_t blocks);
    // Where a batch's n-gram rows come from: uploaded part by part from the feeder's pending
    // gather (each part just before its sub-chunk's ple_inject), or already in ple_rows.
    enum class RowSource : std::uint8_t { Feeder, Resident };

    // Embedding, broadcast and the 48 layers over the uploaded inputs of `chunks`. With
    // RowSource::Feeder the gather of next_tokens for `lane` (when not empty) begins once the last
    // part is uploaded; `streamed` routes the MoE blocks through the windows.
    void enqueue_layers(std::span<const SubChunk> chunks, bool streamed, RowSource rows,
                        std::span<const std::int32_t> next_tokens, const ForwardObserver* observer);
    // enqueue_layers' 48 layers over `chunks`, between the window borrow and return.
    void enqueue_layer_loop(std::span<const SubChunk> chunks, bool streamed, RowSource rows,
                            std::span<const std::int32_t> next_tokens,
                            const ForwardObserver* observer, std::int32_t first_layer = 0,
                            std::int32_t past_layer = kLayers);
    // The super-chunk residual's columns of `chunk`: BF16 [2560,4,T].
    [[nodiscard]] Tensor residual_columns(const SubChunk& chunk) const;
    void gdn_mixer(std::int32_t layer, const SubChunk& chunk);
    void qsa_mixer(std::int32_t layer, const SubChunk& chunk);
    void ple(const SubChunk& chunk);
    void capture_features(std::int32_t layer, const SubChunk& chunk);
    // `streamed`: the super-chunk streams through the windows; `last`: chunk is its last sub-chunk.
    void moe(std::int32_t layer, const SubChunk& chunk, bool streamed, bool last,
             const ForwardObserver* observer);
    [[nodiscard]] bool windowed(std::int32_t tokens) const noexcept;
    // borrow_windows lends 2 window_slots_ consecutive F3 slots out of the cache, starting at the
    // CLOCK hand, or at slot 0 when the run would wrap; it synchronizes stream_ to read the hand.
    // return_windows enqueues on stream_ the return of those slots, empty, to the cache.
    void borrow_windows();
    void return_windows();
    // Enqueues on fill_stream_ the placement and copy of `layer`'s non-resident records into
    // window `buffer`, recording window_planned_ after the placement and window_ready_ after the
    // copy.
    void enqueue_window(std::int32_t layer, std::int32_t buffer);
    // The head over super-chunk residual columns [first, first + width) into logits [248320,width].
    void head(std::int32_t first, std::int32_t width);
    // ops::sample of logits columns [0, rows) with the ingress configurations and slots, then
    // ops::token_logprobs and the egress copy.
    void enqueue_sampling(std::int32_t rows, std::int32_t purpose);
    // Uploads the ingress frame, the round's n-gram rows, then the layers, head and sampling.
    // `observer` (eager rounds only) sees the round's routes and residuals as a prefill does.
    void enqueue_decode(std::int32_t rows, const ForwardObserver* observer);
    void capture_decode_graphs();
    [[nodiscard]] SampledRows sampled_rows(std::int32_t rows) const;
    [[nodiscard]] DecodeIngress& ingress_host() const noexcept;

    const ModelView& model_;
    Dflash* drafter_;
    State& state_;
    PleRowFeeder& feeder_;
    // The expert cache's views: I64 [512,48] tables (column l = layer l) and the pools.
    Tensor expert_table_;
    Tensor host_table_;
    Tensor fill_table_;
    ops::MoeExpertCache cache_;
    ForwardOptions options_;
    cudaStream_t stream_;

    DeviceArena activations_;
    WorkspaceArena work_;
    ChunkStorage chunk_;
    VerifyStorage verify_;
    std::optional<PinnedHostBuffer> verify_ingress_staging_;
    std::optional<PinnedHostBuffer> verify_commit_staging_;
    std::optional<PinnedHostBuffer> spec_ingress_staging_, spec_egress_staging_;
    std::optional<CudaEvent> drafts_ready_;
    std::vector<DecodeGraphDefinition> spec_a_definitions_, spec_b_definitions_;
    std::vector<DecodeGraphExecutable> spec_a_graphs_, spec_b_graphs_;
    bool speculative_pending_          = false;
    std::int32_t pending_verify_width_ = 0;
    std::int32_t pending_verify_rows_  = 0;
    std::int32_t feature_offset_       = 0;
    std::int32_t feature_columns_      = 0;
    // Pinned mirror of ChunkInputs at the same offsets from the activation arena base.
    PinnedHostBuffer input_staging_;
    // Recorded after each chunk's input upload; the host waits on it before rewriting the staging.
    CudaEvent inputs_uploaded_;
    // The decode round's pinned n-gram row frame [12][16][162], ingress frame and sampling egress;
    // fixed addresses captured by the decode graphs. Rewritten only after the previous call that
    // read them synchronized.
    PinnedHostBuffer decode_rows_;
    PinnedHostBuffer ingress_staging_;
    PinnedHostBuffer egress_;
    // decode_graphs_[B - 1] is the round of batch size B (empty with LogitColumns::All).
    std::vector<DecodeGraphDefinition> decode_definitions_;
    std::vector<DecodeGraphExecutable> decode_graphs_;
    CudaEvent replay_start_;
    CudaEvent replay_stop_;
    double last_gather_seconds_ = 0.0;
    float last_replay_ms_       = 0.0F;
    // Each layer's moe_expert_cache_fill runs on fill_stream_ between admitted_ (recorded on
    // stream_ after admit) and filled_, overlapping moe_experts' resident experts; moe_experts
    // waits for filled_ before the deferred ones. Captured into the decode graphs as a fork/join.
    CudaStream fill_stream_;
    CudaEvent admitted_;
    CudaEvent filled_;
    // The two streaming windows are window_slots_ consecutive F3 slots each (0 when none stream),
    // borrowed from the cache for each windowed super-chunk at borrowed_first_ (-1 between
    // super-chunks). Per window, the events order its placement column and records (fill_stream_)
    // against the moe_experts reading them (stream_): planned and ready after the placement and
    // the copy, free once moe_experts has read it; window_start_ (stream_) opens a windowed chunk
    // after all earlier work.
    std::int32_t window_slots_   = 0;
    std::int32_t borrowed_first_ = -1;
    std::array<CudaEvent, 2> window_planned_;
    std::array<CudaEvent, 2> window_ready_;
    std::array<CudaEvent, 2> window_free_;
    CudaEvent window_start_;
};

} // namespace ninfer::targets::qwen3_8_flash_next
