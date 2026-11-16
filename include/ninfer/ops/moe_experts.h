#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_route.h"

#include <cuda_runtime.h> // cudaStream_t

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Qwen3.8-Flash-Next expert geometry: hidden width H, intermediate width I, and the token extent
// one moe_experts call admits.
inline constexpr std::int32_t kMoeHidden       = 2560;
inline constexpr std::int32_t kMoeIntermediate = 640;
inline constexpr std::int32_t kMoeMaxTokens    = 8192;

/** Storage codec of a routed-expert record; bit 0 of the record's expert-table entry. */
enum class MoeExpertCodec : std::uint8_t {
    Nvfp4G16 = 0,
    F3G32    = 1,
};

// Record geometry (see "Routed-expert record" below). A record is a sequence of fixed-size chunk
// records, each covering 32 matrix rows and 128 consecutive K.
inline constexpr std::int64_t kMoeGateUpChunks    = 800; // 40 row tiles x 20 K chunks
inline constexpr std::int64_t kMoeDownChunks      = 400; // 80 row tiles x 5 K chunks
inline constexpr std::int64_t kMoeF3ChunkBytes    = 1664;
inline constexpr std::int64_t kMoeNvfp4ChunkBytes = 2304;
inline constexpr std::int64_t kMoeRecordAlignment = 256;
inline constexpr std::int64_t kMoeF3RecordBytes =
    (kMoeGateUpChunks + kMoeDownChunks) * kMoeF3ChunkBytes;
inline constexpr std::int64_t kMoeNvfp4RecordBytes =
    (kMoeGateUpChunks + kMoeDownChunks) * kMoeNvfp4ChunkBytes;
static_assert(kMoeF3RecordBytes == 1996800 && kMoeNvfp4RecordBytes == 2764800);

[[nodiscard]] constexpr std::int64_t moe_expert_chunk_bytes(MoeExpertCodec codec) {
    return codec == MoeExpertCodec::F3G32 ? kMoeF3ChunkBytes : kMoeNvfp4ChunkBytes;
}

[[nodiscard]] constexpr std::int64_t moe_expert_record_bytes(MoeExpertCodec codec) {
    return codec == MoeExpertCodec::F3G32 ? kMoeF3RecordBytes : kMoeNvfp4RecordBytes;
}

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

/** Workspace bytes of moe_experts for every T in [1, max_tokens]; max_tokens in [1,8192]. */
[[nodiscard]] std::size_t moe_experts_workspace_capacity_bytes(std::int32_t max_tokens);

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
 * outputs); shared_scale FP32 [T]; y BF16 [2560,T]; 1 <= T <= 8192. All are contiguous, 16-byte
 * aligned, and y is disjoint from every input. The ten ids of a token are distinct and in [0,512),
 * as moe_route produces them; violating this is a precondition violation whose result is
 * unspecified (an id outside [0,512) is clamped, so the Op still touches only its documented
 * buffers).
 *
 * Numeric: the oracle evaluates the formula in FP64 from the represented BF16 x, the decoded
 * weights (codes through the book or E2M1 code, exact E4M3 and FP32 scales) and the FP32 route
 * weights and shared scale. Implementation profile (A16): decoded code times E4M3 scale is exact in
 * BF16 and feeds BF16 tensor-core products with FP32 accumulation; the matrix scales, SiLU and the
 * gate/up product are FP32; that product enters the down projection rounded once to BF16; each
 * scaled down output is FP32, and the eleven per-token contributions are summed in slot order (j =
 * 0..9, then the shared expert) and rounded once to BF16. No float atomics: y is bit-identical for
 * identical inputs and T; the accumulation association may differ between extents.
 *
 * Records are immutable while enqueued work may read them; the Op writes only y and its workspace.
 *
 * Deferral (optional): with a deferral, every routed expert e with slots[e] != 0 that a token
 * routes to is read from slots[e] instead of expert_table[e], and only after `ready`: the shared
 * and the other routed experts are computed first on `stream`, which then waits for `ready` and
 * computes the deferred experts. y is bit-identical to the call without a deferral on the same
 * record bytes, since every assignment's contribution and the merge order are unchanged.
 *
 * Workspace: moe_experts_workspace_capacity_bytes(T) bytes from a 256-byte aligned arena cursor;
 * the allocations are released on return.
 *
 * Execution: asynchronous on stream with no host synchronization or readback. The tables, scales
 * and records are read on the device at execution, so a captured CUDA Graph is valid for a fixed T
 * and fixed tensor addresses (and, with a deferral, a fixed slots column whose contents may change)
 * while table, scale and record contents change between replays.
 * Invalid shapes, dtypes, alignment or aliasing throw std::invalid_argument before any work is
 * queued.
 */
void moe_experts(const Tensor& x, const Tensor& expert_ids, const Tensor& route_weights,
                 const Tensor& shared_scale, const MoeExpertBank& bank, Tensor& y,
                 WorkspaceArena& workspace, cudaStream_t stream,
                 const MoeExpertsDeferral* deferral = nullptr);

} // namespace ninfer::ops
