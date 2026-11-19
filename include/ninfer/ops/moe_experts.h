#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_expert_record.h"
#include "ninfer/ops/moe_route.h"

#include <cuda_runtime.h> // cudaStream_t

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// The token extent one moe_experts call admits (the expert geometry and record sizes are in
// ninfer/ops/moe_expert_record.h): a whole 16,384-token prefill super-chunk.
inline constexpr std::int32_t kMoeMaxTokens = 16384;

/**
 * Activation precision profile of one moe_experts call, chosen by the caller. Both profiles
 * evaluate the same mathematical formula and are checked against the same FP64 oracle; they differ
 * only in how activations enter the tensor-core products (see "Numeric" in moe_experts).
 *
 *   A16  BF16 activations: the decode, verification and default prefill profile.
 *   A8   E4M3 activations with one FP32 scale per row (moe_activation_quantize's codec): the
 *        prefill profile shared with the R9700 expert helper, which applies the same codec, so an
 *        expert's products agree across the two cards up to FP32 accumulation order. It doubles
 *        the tensor-core rate of F3G32 experts (one E4M3 block scale per 32 K fits one m16n8k32
 *        step) and keeps that of NVFP4G16 experts (a scale per 16 K needs m16n8k16 steps). Valid
 *        for every T; its single 64-token tile is sized for prefill sub-chunks.
 */
enum class MoeActivationProfile : std::uint8_t { A16, A8 };

/**
 * Routed-expert record
 *
 * One routed expert e holds three matrices: gate G and up U, each [640,2560] (row = intermediate
 * index, K = hidden index), and down D [2560,640] (row = hidden index, K = intermediate index). Its
 * record is a 256-byte aligned byte string of moe_expert_record_bytes(codec) bytes (F3G32
 * 1,996,800, NVFP4G16 2,764,800) made of chunk records of B = moe_expert_chunk_bytes(codec) bytes
 * (F3G32 1664, NVFP4G16 2304). Offsets are bytes from the record address; multi-byte words are
 * little-endian.
 *
 *   gate|up plane  chunk (r, q) at (20 r + q) B,         r in [0,40), q in [0,20)
 *   down plane     chunk (r, q) at (800 + 5 r + q) B,    r in [0,80), q in [0,5)
 *
 * Chunk (r, q) covers 32 local rows rho in [0,32) and K = 128 q + kappa, kappa in [0,128). In the
 * gate|up plane local row rho < 16 is gate row 16 r + rho and rho >= 16 is up row 16 r + rho - 16;
 * in the down plane local row rho is down row 32 r + rho. Within a chunk, lane L in [0,32), with
 * g = L / 4 and c = L % 4, owns the four local rows rho_k = g + 8 k (k in [0,4)) and the 32
 * consecutive K offsets kappa = 32 c + i (i in [0,32)).
 *
 * F3G32 chunk. Each (rho_k, c) is one unit: 32 codes kappa_0..kappa_31 (3-bit indices into the
 * matrix's codebook, code i at K offset 32 c + i) and one E4M3 scale.
 *   byte 4 L + k                 the unit's E4M3 scale (bytes [0,128))
 *   word at 128 + 16 (32 j + L) + 4 k, j in [0,3)
 *                                the unit's code word w_j: nibble n (bits 4n..4n+3) holds code
 *                                kappa_{8j+n} in its low three bits, and bit j of code kappa_{24+n}
 *                                in its high bit
 * The value is book_m[kappa] times the E4M3 scale, where book_m is the matrix's codebook: eight
 * E2M1 nibbles, entry v in bits 4v..4v+3 (sign in the nibble's high bit).
 *
 * NVFP4G16 chunk. Every 16 consecutive K of a row share one E4M3 scale.
 *   byte 8 L + 2 k + h           E4M3 scale of row rho_k, K offsets 32 c + 16 h .. +15 (bytes
 *                                [0,256))
 *   byte 256 + 16 (32 j + L) + 4 k + b, j in [0,4), b in [0,4)
 *                                E2M1 codes of row rho_k at K offset 32 c + 8 j + 2 b (low
 *                                nibble) and 32 c + 8 j + 2 b + 1 (high nibble)
 * The value is the E2M1 code times its E4M3 scale.
 *
 * Every decoded matrix value is multiplied by the expert's FP32 matrix scale for that matrix.
 */

/**
 * Weights of one layer's experts. The routed records are addressed through a device table; the
 * shared expert is one NVFP4G16 record in the routed record layout, always in device memory.
 *
 *   expert_table    I64 [512] on the device, the U64 bit pattern addr | codec: the 256-byte
 *                   aligned address of expert e's record (device memory or device-mapped pinned
 *                   host memory, read the same way) with its MoeExpertCodec in bit 0. Entries of
 *                   experts no token routes to are not read.
 *   matrix_scales   FP32 [512,3] (Tensor ne {3,512}): element 3 e + m is expert e's scale of
 *                   matrix m (0 gate, 1 up, 2 down).
 *   f3_books        the layer's F3G32 codebooks of gate, up and down: eight E2M1 nibbles each,
 *                   passed by value (a captured CUDA Graph keeps the books of its capture).
 *   shared_record   U8 [2764800] (kMoeNvfp4RecordBytes), 256-byte aligned device memory: the
 *                   shared expert's NVFP4G16 record, laid out exactly as a routed NVFP4G16 record.
 *   shared_matrix_scales
 *                   FP32 [3]: the shared expert's scales of gate, up and down.
 */
struct MoeExpertBank {
    Tensor expert_table;
    Tensor matrix_scales;
    std::array<std::uint32_t, 3> f3_books{};
    Tensor shared_record;
    Tensor shared_matrix_scales;
};

/**
 * Routed records still being written when moe_experts is enqueued, typically by
 * moe_expert_cache_fill on a second stream:
 *
 *   slots  I64 [512] on the device: for expert e, 0 when its record is read through the expert
 *          table, else the untagged 256-byte aligned VRAM address its record is copied to. The
 *          record keeps the codec of expert_table[e] (bit 0), and slots[e] is disjoint from every
 *          input, output and workspace of the call.
 *   ready  an event the caller records after every nonzero slot holds its complete record.
 *
 * slots is read on the device by the call's first kernel, so its contents are complete in
 * `stream` order when moe_experts is enqueued (ready covers only the records); ready is waited on
 * by `stream` when moe_experts is enqueued, so the caller records it beforehand (host order).
 * Neither the slots tensor nor the deferred records it names may change until the call completes.
 * Under CUDA Graph capture, ready is recorded inside the same capture on a stream forked from
 * `stream`.
 */
struct MoeExpertsDeferral {
    Tensor slots;
    cudaEvent_t ready = nullptr;
};

/**
 * Workspace bytes of moe_experts and moe_experts_local_partial under `profile` for every T in
 * [1, max_tokens]; max_tokens in [1,16384], else std::invalid_argument.
 */
[[nodiscard]] std::size_t moe_experts_workspace_capacity_bytes(std::int32_t max_tokens,
                                                               MoeActivationProfile profile);

/**
 * Op: moe_activation_quantize
 *
 * The A8 activation codec of moe_experts (MoeActivationProfile::A8) and of the R9700 expert
 * helper's prefill, one row at a time. For each token t with the represented BF16 row x[:,t] of
 * K = 2560 values (all in FP32, IEEE round-to-nearest-even):
 *
 *   scale[t]   = max_k |x[k,t]| / 448                  (a single FP32 division)
 *   codes[k,t] = E4M3FN_satfinite_rn(x[k,t] / scale[t]) (FP32 division, then round to nearest
 *                                                        even, saturating to +-448; never NaN)
 *
 * except that an all-zero row (max 0, including -0 values) has scale +0 and every code +0. The
 * represented value is decode(codes[k,t]) * scale[t]. Within a nonzero row a value that rounds to
 * zero keeps its sign (0x80 for a negative one), and E4M3 subnormal codes are produced. This is the
 * cross-card contract: codes and scales are bit-exact functions of the BF16 bits. It differs from
 * A8Activation's codec (ninfer/ops/a8_activation.h), which multiplies by the FP32 reciprocal of the
 * scale and so can round differently.
 *
 * Shapes: x BF16 [2560,T], codes FP8_E4M3FN [2560,T], scales FP32 [T], 1 <= T <= 16384; all
 * contiguous and 16-byte aligned, the outputs disjoint from x and from each other. x must be
 * finite. Asynchronous on stream, one warp per row, graph-capturable for fixed views, no workspace.
 * Invalid shapes, dtypes, alignment or aliasing throw std::invalid_argument before any work is
 * queued.
 */
void moe_activation_quantize(const Tensor& x, Tensor& codes, Tensor& scales, cudaStream_t stream);

/**
 * Op: moe_experts
 *
 * Math, for each token t with x_t = x[:,t] and e_j = expert_ids[j,t]:
 *
 *   E(x)    = D (SiLU(G x) * U x)                     (SiLU(z) = z / (1 + exp(-z)), * elementwise)
 *   y[:,t]  = sum_{j<10} route_weights[j,t] E_{e_j}(x_t) + shared_scale[t] E_shared(x_t)
 *
 * where E_e uses routed expert e's decoded matrices and E_shared the shared record's, both as
 * "Routed-expert record" describes (the shared expert with shared_matrix_scales).
 *
 * Logical shapes: x BF16 [2560,T]; expert_ids I32 [10,T] and route_weights FP32 [10,T] (moe_route's
 * outputs); shared_scale FP32 [T]; y BF16 [2560,T]; 1 <= T <= 16384. All are contiguous, 16-byte
 * aligned, and y is disjoint from every input. The ten ids of a token are distinct and in [0,512),
 * as moe_route produces them; violating this is a precondition violation whose result is
 * unspecified (an id outside [0,512) is clamped, so the Op still touches only its documented
 * buffers).
 *
 * Numeric: the oracle evaluates the formula in FP64 from the represented BF16 x, the decoded
 * weights (codes through the book or E2M1 code, exact E4M3 and FP32 scales) and the FP32 route
 * weights and shared scale. The two implementation profiles (`profile`) share every step except
 * how activations enter the products:
 *
 *   A16  decoded code times E4M3 scale is exact in BF16 and feeds BF16 tensor-core products of BF16
 *        activations with FP32 accumulation.
 *   A8   every gate|up input row x[:,t] and every down input row (the BF16 gate/up product of one
 *        assignment, shared expert included) is encoded by moe_activation_quantize's codec (FP32
 *        row scale, E4M3 codes). Weight codes expand exactly to E4M3 (every E2M1 value is one) and
 *        multiply the activation codes in E4M3 tensor-core products with FP32 accumulation, one
 *        product per weight block (an m16n8k32 step per F3G32 unit of 32 K, an m16n8k16 step per
 *        NVFP4G16 group of 16 K); each block's FP32 partial times its E4M3 block scale is added
 *        to the FP32 accumulator in one fused multiply-add, in ascending K order, and the
 *        activation row scale multiplies the accumulated sum. Block scales are never folded into
 *        codes.
 *
 * In both, the matrix scales, SiLU and the gate/up product are FP32; that product enters the down
 * projection rounded once to BF16; each scaled down output is FP32, and the eleven per-token
 * contributions are summed in slot order (j = 0..9, then the shared expert) and rounded once to
 * BF16. No float atomics: y is bit-identical for identical inputs, profile and T; the accumulation
 * association may differ between extents.
 *
 * Records are immutable while enqueued work may read them; the Op writes only y and its workspace.
 *
 * Deferral (optional): with a deferral, every routed expert e with slots[e] != 0 that a token
 * routes to is read from slots[e] instead of expert_table[e], and only after `ready`: the shared
 * and the other routed experts are computed first on `stream`, which then waits for `ready` and
 * computes the deferred experts. y is bit-identical to the call without a deferral on the same
 * record bytes, since every assignment's contribution and the merge order are unchanged.
 *
 * Workspace: moe_experts_workspace_capacity_bytes(T, profile) bytes from a 256-byte aligned arena
 * cursor; the allocations are released on return.
 *
 * Execution: asynchronous on stream with no host synchronization or readback. The tables, scales
 * and records are read on the device at execution, so a captured CUDA Graph is valid for a fixed T
 * and fixed tensor addresses (and, with a deferral, a fixed slots column whose contents may change)
 * while table, scale and record contents change between replays.
 * Invalid shapes, dtypes, alignment or aliasing throw std::invalid_argument before any work is
 * queued.
 */
void moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& route_weights,
                 const Tensor& shared_scale, const MoeExpertBank& bank,
                 MoeActivationProfile profile, Tensor& y, WorkspaceArena& workspace,
                 cudaStream_t stream, const MoeExpertsDeferral* deferral = nullptr);

/**
 * Op: moe_experts_local_partial
 *
 * The RTX 5090 side of the expert-helper split
 * (docs/maintainer/qwen3.8-flash-next-expert-helper.md): moe_experts restricted to the local
 * experts, without output rounding. With L the experts whose bit is set in local_experts (I32 [16]
 * holding U32 bit patterns, bit e % 32 of word e / 32; see ninfer/ops/moe_helper_contract.h):
 *
 *   partial[:,t] = sum_{j<10, expert_ids[j,t] in L} route_weights[j,t] E_{expert_ids[j,t]}(x_t)
 *                  + shared_scale[t] E_shared(x_t)
 *
 * Every other routed assignment is excluded: it is not computed and its record is not read, so
 * the table entries of non-local experts may name any value. The shared expert is always
 * included, also when no routed expert of a token is local. Adding moe_helper_experts' output for
 * the remaining assignments gives moe_experts' sum (moe_expert_partial_join); only the reduction
 * association differs.
 *
 * Shapes, operands, profile, deferral, workspace (moe_experts_workspace_capacity_bytes) and
 * execution are as for moe_experts, with partial FP32 [2560,T] in place of y: contiguous, 16-byte
 * aligned and disjoint from every input. local_experts is contiguous I32 [16], 16-byte aligned,
 * read on the device when the call executes (a captured graph stays valid while it changes between
 * calls); it must not change while the call runs, since the grouping and the final sum each read
 * it.
 *
 * Numeric: moe_experts' implementation profiles, except that each token's sum over its included
 * slots in slot order (local routed j ascending, then the shared expert) is stored as FP32, not
 * rounded to BF16. Deterministic for identical inputs and T.
 */
void moe_experts_local_partial(const Tensor& x, const Tensor& expert_ids,
                               const Tensor& route_weights, const Tensor& shared_scale,
                               const MoeExpertBank& bank, MoeActivationProfile profile,
                               const Tensor& local_experts, Tensor& partial,
                               WorkspaceArena& workspace, cudaStream_t stream,
                               const MoeExpertsDeferral* deferral = nullptr);

} // namespace ninfer::ops
