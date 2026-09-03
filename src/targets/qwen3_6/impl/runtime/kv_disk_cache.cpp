#include "targets/qwen3_6/impl/runtime/kv_disk_cache.h"

#include "core/device.h"

#include <zstd.h>

#if defined(__x86_64__)
#include <nmmintrin.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/uio.h>
#include <unistd.h>
#include <unordered_set>
#include <utility>

namespace ninfer::targets::qwen3_6::detail {
namespace {

std::mutex g_location_mu;
std::unordered_set<std::string> g_locations;

struct OutBuf {
    std::vector<std::uint8_t> bytes;
    void u8(std::uint8_t v) { bytes.push_back(v); }
    void u16(std::uint16_t v) {
        u8(static_cast<std::uint8_t>(v));
        u8(static_cast<std::uint8_t>(v >> 8));
    }
    void u32(std::uint32_t v) {
        u8(static_cast<std::uint8_t>(v));
        u8(static_cast<std::uint8_t>(v >> 8));
        u8(static_cast<std::uint8_t>(v >> 16));
        u8(static_cast<std::uint8_t>(v >> 24));
    }
    void i32(std::int32_t v) { u32(static_cast<std::uint32_t>(v)); }
    void u64(std::uint64_t v) {
        for (int s = 0; s < 64; s += 8) { u8(static_cast<std::uint8_t>(v >> s)); }
    }
    void i64(std::int64_t v) { u64(static_cast<std::uint64_t>(v)); }
    void raw(const void* data, std::size_t n) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + n);
    }
    void str(const std::string& s) {
        u32(static_cast<std::uint32_t>(s.size()));
        raw(s.data(), s.size());
    }
};

struct InBuf {
    const std::uint8_t* p   = nullptr;
    const std::uint8_t* end = nullptr;
    [[nodiscard]] std::size_t remain() const { return static_cast<std::size_t>(end - p); }
    [[nodiscard]] std::uint8_t u8() {
        if (p >= end) { throw std::runtime_error("KV disk buffer underflow"); }
        return *p++;
    }
    [[nodiscard]] std::uint16_t u16() { return static_cast<std::uint16_t>(u8() | (u8() << 8)); }
    [[nodiscard]] std::uint32_t u32() {
        const std::uint32_t a = u8(), b = u8(), c = u8(), d = u8();
        return a | (b << 8) | (c << 16) | (d << 24);
    }
    [[nodiscard]] std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t v = 0;
        for (int s = 0; s < 64; s += 8) { v |= static_cast<std::uint64_t>(u8()) << s; }
        return v;
    }
    [[nodiscard]] std::int64_t i64() { return static_cast<std::int64_t>(u64()); }
    void raw(void* dst, std::size_t n) {
        if (remain() < n) { throw std::runtime_error("KV disk buffer underflow"); }
        std::memcpy(dst, p, n);
        p += n;
    }
    [[nodiscard]] std::string str() {
        const auto n = u32();
        if (remain() < n) { throw std::runtime_error("KV disk buffer underflow"); }
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
    void skip(std::size_t n) {
        if (remain() < n) { throw std::runtime_error("KV disk buffer underflow"); }
        p += n;
    }
};

[[nodiscard]] const char* kind_dir(DiskObjectKind kind) {
    switch (kind) {
    case DiskObjectKind::Main:
        return "main";
    case DiskObjectKind::Backend:
        return "backend";
    case DiskObjectKind::State:
        return "state";
    case DiskObjectKind::Ledger:
        return "ledger";
    case DiskObjectKind::Identity:
        return "identity";
    }
    return "state";
}

[[nodiscard]] constexpr std::size_t kind_index(DiskObjectKind kind) {
    return static_cast<std::size_t>(kind);
}

struct PackedMapEntry {
    std::uint64_t id = 0;
    DiskObjectKind kind = DiskObjectKind::Main;
    std::uint32_t segment = 0;
    std::uint64_t offset = 0;
    std::uint64_t extent_bytes = 0;
    std::uint64_t stored_bytes = 0;
    std::uint64_t logical_bytes = 0;
    std::uint32_t record_crc32c = 0;
};

// frame_bytes includes the u32 length field and excludes the trailing frame CRC.
// The canonical Put payload is id/kind/segment/offset/extent/stored/logical/record_crc.
inline constexpr std::uint8_t kPackMapTypePut = 1;
inline constexpr std::uint32_t kPackMapPutBytes   = 52;
inline constexpr std::uint32_t kPackMapFrameBytes = 4 + 4 + kPackMapPutBytes;
inline constexpr std::uint32_t kPackMapBaseEndianLittle = 1;
inline constexpr std::uint32_t kPackMapBaseHeaderBytes = 8 + 4 + 4 + 8 + 8 + 8 + 8;

[[nodiscard]] std::uint32_t crc32c_update_software(
    std::uint32_t crc, std::span<const std::uint8_t> bytes) noexcept {
    for (std::uint8_t byte : bytes) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0x82F63B78U & static_cast<std::uint32_t>(-(crc & 1U)));
        }
    }
    return crc;
}

#if defined(__x86_64__)
__attribute__((target("sse4.2")))
[[nodiscard]] std::uint32_t crc32c_update_hardware(
    std::uint32_t crc, std::span<const std::uint8_t> bytes) noexcept {
    const auto* data = bytes.data();
    std::size_t size = bytes.size();
    while (size >= sizeof(std::uint64_t)) {
        std::uint64_t word = 0;
        std::memcpy(&word, data, sizeof(word));
        crc = static_cast<std::uint32_t>(_mm_crc32_u64(crc, word));
        data += sizeof(word);
        size -= sizeof(word);
    }
    if (size >= sizeof(std::uint32_t)) {
        std::uint32_t word = 0;
        std::memcpy(&word, data, sizeof(word));
        crc = _mm_crc32_u32(crc, word);
        data += sizeof(word);
        size -= sizeof(word);
    }
    if (size >= sizeof(std::uint16_t)) {
        std::uint16_t word = 0;
        std::memcpy(&word, data, sizeof(word));
        crc = _mm_crc32_u16(crc, word);
        data += sizeof(word);
        size -= sizeof(word);
    }
    if (size != 0) { crc = _mm_crc32_u8(crc, *data); }
    return crc;
}
#endif

[[nodiscard]] std::uint32_t crc32c_update(std::uint32_t crc,
                                           std::span<const std::uint8_t> bytes) noexcept {
#if defined(__x86_64__)
    static const bool hardware = __builtin_cpu_supports("sse4.2");
    if (hardware) { return crc32c_update_hardware(crc, bytes); }
#endif
    return crc32c_update_software(crc, bytes);
}

[[nodiscard]] std::uint32_t crc32c_combine(std::uint32_t left, std::uint32_t right,
                                            std::uint64_t right_bytes) noexcept;

#if defined(__x86_64__)
__attribute__((target("sse4.2")))
[[nodiscard]] std::uint32_t crc32c_three_way_hardware(
    std::span<const std::uint8_t> bytes) noexcept {
    const std::size_t block = (bytes.size() / 3) & ~std::size_t{7};
    const auto* first = bytes.data();
    const auto* second = first + block;
    const auto* third = second + block;
    std::uint32_t crc_first = ~0U;
    std::uint32_t crc_second = ~0U;
    std::uint32_t crc_third = ~0U;
    for (std::size_t off = 0; off < block; off += sizeof(std::uint64_t)) {
        std::uint64_t a = 0, b = 0, c = 0;
        std::memcpy(&a, first + off, sizeof(a));
        std::memcpy(&b, second + off, sizeof(b));
        std::memcpy(&c, third + off, sizeof(c));
        crc_first = static_cast<std::uint32_t>(_mm_crc32_u64(crc_first, a));
        crc_second = static_cast<std::uint32_t>(_mm_crc32_u64(crc_second, b));
        crc_third = static_cast<std::uint32_t>(_mm_crc32_u64(crc_third, c));
    }
    crc_third = crc32c_update_hardware(
        crc_third, bytes.subspan(3 * block, bytes.size() - 3 * block));
    const std::uint32_t first_second =
        crc32c_combine(~crc_first, ~crc_second, block);
    return crc32c_combine(first_second, ~crc_third, bytes.size() - 2 * block);
}
#endif

[[nodiscard]] std::uint32_t crc32c(std::span<const std::uint8_t> bytes) noexcept {
#if defined(__x86_64__)
    constexpr std::size_t kParallelCrcMinBytes = 512ULL << 10;
    static const bool hardware = __builtin_cpu_supports("sse4.2");
    if (hardware && bytes.size() >= kParallelCrcMinBytes) {
        return crc32c_three_way_hardware(bytes);
    }
#endif
    return ~crc32c_update(~0U, bytes);
}

[[nodiscard]] std::uint32_t crc32c_matrix_times(const std::uint32_t* matrix,
                                                 std::uint32_t vector) noexcept {
    std::uint32_t result = 0;
    while (vector != 0) {
        if ((vector & 1U) != 0) { result ^= *matrix; }
        vector >>= 1;
        ++matrix;
    }
    return result;
}

void crc32c_matrix_square(std::uint32_t* square, const std::uint32_t* matrix) noexcept {
    for (int bit = 0; bit < 32; ++bit) {
        square[bit] = crc32c_matrix_times(matrix, matrix[bit]);
    }
}

// Combine two finalized CRC-32C values without reading the right-hand bytes a
// second time. This is the reflected-polynomial concatenation used by the
// on-disk record CRC; payload CRC remains independently stored in the header.
[[nodiscard]] std::uint32_t crc32c_combine(std::uint32_t left, std::uint32_t right,
                                            std::uint64_t right_bytes) noexcept {
    if (right_bytes == 0) { return left; }
    std::uint32_t odd[32]{};
    std::uint32_t even[32]{};
    odd[0] = 0x82F63B78U;
    std::uint32_t row = 1;
    for (int bit = 1; bit < 32; ++bit) {
        odd[bit] = row;
        row <<= 1;
    }
    crc32c_matrix_square(even, odd);
    crc32c_matrix_square(odd, even);
    do {
        crc32c_matrix_square(even, odd);
        if ((right_bytes & 1U) != 0) { left = crc32c_matrix_times(even, left); }
        right_bytes >>= 1;
        if (right_bytes == 0) { break; }
        crc32c_matrix_square(odd, even);
        if ((right_bytes & 1U) != 0) { left = crc32c_matrix_times(odd, left); }
        right_bytes >>= 1;
    } while (right_bytes != 0);
    return left ^ right;
}

[[nodiscard]] std::uint32_t crc32c_parts(
    std::span<const std::pair<const void*, std::uint64_t>> parts) noexcept {
    std::uint32_t crc = ~0U;
    for (const auto& [data, bytes] : parts) {
        if (data == nullptr || bytes == 0) { continue; }
        crc = crc32c_update(crc, std::span<const std::uint8_t>(
                                  static_cast<const std::uint8_t*>(data),
                                  static_cast<std::size_t>(bytes)));
    }
    return ~crc;
}

ssize_t pwritev_with_test_limit(int fd, iovec* iov, int count, off_t offset,
                                std::size_t max_bytes) {
    if (max_bytes == 0) { return ::pwritev(fd, iov, count, offset); }
    std::vector<iovec> limited;
    limited.reserve(static_cast<std::size_t>(count));
    std::size_t remaining = max_bytes;
    for (int i = 0; i < count && remaining != 0; ++i) {
        iovec part = iov[i];
        if (part.iov_len > remaining) { part.iov_len = remaining; }
        limited.push_back(part);
        remaining -= part.iov_len;
    }
    if (limited.empty()) { throw std::logic_error("partial pwritev test limit is empty"); }
    return ::pwritev(fd, limited.data(), static_cast<int>(limited.size()), offset);
}

[[nodiscard]] std::vector<std::uint8_t> encode_packset(std::uint64_t generation,
                                                         std::uint64_t next_object_id) {
    OutBuf w;
    w.raw(kDiskPackSetMagic, 8);
    w.u32(kDiskFormatVersion);
    w.u64(generation);
    w.u64(next_object_id);
    const std::uint32_t crc = crc32c(w.bytes);
    w.u32(crc);
    return std::move(w.bytes);
}

[[nodiscard]] bool decode_packset(std::span<const std::uint8_t> bytes,
                                  std::uint64_t& generation,
                                  std::uint64_t& next_object_id) {
    if (bytes.size() != 32 || std::memcmp(bytes.data(), kDiskPackSetMagic, 8) != 0) {
        return false;
    }
    if (crc32c(bytes.first(28)) !=
        (static_cast<std::uint32_t>(bytes[28]) |
         (static_cast<std::uint32_t>(bytes[29]) << 8) |
         (static_cast<std::uint32_t>(bytes[30]) << 16) |
         (static_cast<std::uint32_t>(bytes[31]) << 24))) {
        return false;
    }
    try {
        InBuf in{bytes.data() + 8, bytes.data() + 28};
        if (in.u32() != kDiskFormatVersion) { return false; }
        generation = in.u64();
        next_object_id = in.u64();
        return generation != 0 && next_object_id != 0;
    } catch (...) { return false; }
}

[[nodiscard]] std::vector<std::uint8_t> encode_pack_map_frame(const PackedMapEntry& entry) {
    OutBuf w;
    w.u32(kPackMapFrameBytes);
    w.u8(kPackMapTypePut);
    w.u8(0); w.u8(0); w.u8(0);
    w.u64(entry.id);
    w.u8(static_cast<std::uint8_t>(entry.kind));
    w.u8(0); w.u8(0); w.u8(0);
    w.u32(entry.segment);
    w.u64(entry.offset);
    w.u64(entry.extent_bytes);
    w.u64(entry.stored_bytes);
    w.u64(entry.logical_bytes);
    w.u32(entry.record_crc32c);
    const std::uint32_t crc = crc32c(w.bytes);
    w.u32(crc);
    return std::move(w.bytes);
}

[[nodiscard]] std::vector<std::uint8_t> encode_pack_map_base(
    std::uint64_t generation, std::uint64_t next_id,
    std::vector<PackedMapEntry> entries) {
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    OutBuf w;
    w.raw(kDiskObjectMapMagic, 8);
    w.u32(kDiskFormatVersion);
    w.u32(kPackMapBaseEndianLittle);
    w.u64(generation);
    w.u64(next_id);
    w.u64(entries.size());
    w.u64(entries.size() * static_cast<std::uint64_t>(kPackMapPutBytes));
    for (const auto& entry : entries) {
        w.u64(entry.id); w.u8(static_cast<std::uint8_t>(entry.kind));
        w.u8(0); w.u8(0); w.u8(0); w.u32(entry.segment); w.u64(entry.offset);
        w.u64(entry.extent_bytes); w.u64(entry.stored_bytes); w.u64(entry.logical_bytes);
        w.u32(entry.record_crc32c);
    }
    w.u32(crc32c(w.bytes));
    return std::move(w.bytes);
}

[[nodiscard]] bool valid_checkpoint_kind(std::uint32_t kind) noexcept {
    return kind <= static_cast<std::uint32_t>(ContextCheckpointKind::OnDemand);
}

[[nodiscard]] bool valid_rewrite_kind(std::uint8_t kind) noexcept {
    return kind <= static_cast<std::uint8_t>(RewriteCheckpointKind::ResponseReplay);
}

[[nodiscard]] bool valid_state_kind(std::uint8_t kind) {
    return kind >= static_cast<std::uint8_t>(DiskStateKind::CurrentGdn) &&
           kind <= static_cast<std::uint8_t>(DiskStateKind::DflashLadder);
}

[[nodiscard]] bool payload_fits(std::uint64_t file_bytes, std::uint64_t header,
                                 std::uint64_t payload) noexcept {
    return file_bytes >= header && payload <= file_bytes - header;
}

constexpr std::uint64_t kDiskHostFileMaxBytes = 64ULL << 20;
constexpr std::uint32_t kDiskMetaMaxPageIds   = 1u << 16;

[[nodiscard]] std::size_t aligned_page_file_bytes(std::size_t payload) {
    const std::size_t bytes = kDiskPageHeaderBytes + payload;
    return (bytes + kDiskPageIoAlignment - 1) &
           ~(static_cast<std::size_t>(kDiskPageIoAlignment) - 1);
}

[[nodiscard]] std::size_t aligned_state_file_bytes(std::size_t payload) {
    const std::size_t bytes = kDiskStatePayloadOffset + payload;
    return (bytes + kDiskPageIoAlignment - 1) &
           ~(static_cast<std::size_t>(kDiskPageIoAlignment) - 1);
}

[[nodiscard]] bool codec_payload_ok(DiskCodec codec, std::uint64_t unc, std::uint64_t cmp,
                                    std::uint64_t max_unc = kDiskHostFileMaxBytes) noexcept {
    if (unc == 0 || cmp == 0) { return false; }
    if (unc > max_unc) { return false; }
    if (codec == DiskCodec::Raw) { return unc == cmp; }
    if (codec != DiskCodec::Zstd) { return false; }
    if (unc > std::numeric_limits<std::size_t>::max()) { return false; }
    return cmp <= ZSTD_compressBound(static_cast<std::size_t>(unc));
}

[[nodiscard]] bool prefix_extension(std::span<const TokenId> captured,
                                    std::span<const TokenId> committed, std::uint32_t committed_e) {
    const std::size_t need = static_cast<std::size_t>(committed_e) + 1U;
    if (committed.size() < need || captured.size() < need) { return false; }
    return std::equal(committed.begin(), committed.begin() + static_cast<std::ptrdiff_t>(need),
                      captured.begin());
}

[[nodiscard]] std::uint32_t longest_ledger_prefix(std::span<const TokenId> a,
                                                  std::span<const TokenId> b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i       = 0;
    while (i < n && a[i] == b[i]) { ++i; }
    return static_cast<std::uint32_t>(i);
}

void append_plane_fingerprint(OutBuf& w, const Tensor& plane, PagedKVPlaneOrder order) {
    w.u8(static_cast<std::uint8_t>(plane.dtype));
    w.u8(static_cast<std::uint8_t>(order));
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    for (int i = 0; i < 4; ++i) { w.i32(plane.ne[i]); }
    for (int i = 0; i < 4; ++i) { w.i64(plane.nb[i]); }
}

[[nodiscard]] std::vector<std::uint8_t> encode_fingerprint(const DiskFingerprint& fp) {
    OutBuf w;
    w.raw(kDiskFingerprintMagic, 8);
    w.u32(kDiskFormatVersion);
    w.str(fp.model_id);
    w.str(fp.weights_id);
    w.u8(static_cast<std::uint8_t>(fp.kv_cache));
    w.u8(static_cast<std::uint8_t>(fp.speculative));
    w.u32(fp.page_size);
    w.u32(fp.text_plane_count);
    w.u32(fp.backend_plane_count);
    w.u64(fp.gdn_conv_bytes);
    w.u64(fp.gdn_recurrent_bytes);
    w.u64(fp.cyclic_lane_bytes);
    w.u32(fp.cyclic_layers);
    w.u32(fp.cyclic_capacity);
    w.u32(fp.cyclic_padded);
    w.i32(fp.cyclic_kv_heads);
    w.i32(fp.cyclic_head_dim);
    w.i32(fp.cyclic_lane_capacity);
    w.u32(fp.snapshot_version);
    w.raw(fp.text_plane_bytes.data(), fp.text_plane_bytes.size());
    w.raw(fp.backend_plane_bytes.data(), fp.backend_plane_bytes.size());
    return std::move(w.bytes);
}

[[nodiscard]] DiskFingerprint decode_fingerprint(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 8 || std::memcmp(bytes.data(), kDiskFingerprintMagic, 8) != 0) {
        throw std::runtime_error("KV disk FINGERPRINT magic mismatch");
    }
    InBuf r{bytes.data() + 8, bytes.data() + bytes.size()};
    if (r.u32() != kDiskFormatVersion) {
        throw std::runtime_error("KV disk FINGERPRINT version mismatch");
    }
    DiskFingerprint fp;
    fp.model_id             = r.str();
    fp.weights_id           = r.str();
    fp.kv_cache             = static_cast<KvCacheStorage>(r.u8());
    fp.speculative          = static_cast<SpeculativeBackend>(r.u8());
    fp.page_size            = r.u32();
    fp.text_plane_count     = r.u32();
    fp.backend_plane_count  = r.u32();
    fp.gdn_conv_bytes       = r.u64();
    fp.gdn_recurrent_bytes  = r.u64();
    fp.cyclic_lane_bytes    = r.u64();
    fp.cyclic_layers        = r.u32();
    fp.cyclic_capacity      = r.u32();
    fp.cyclic_padded        = r.u32();
    fp.cyclic_kv_heads      = r.i32();
    fp.cyclic_head_dim      = r.i32();
    fp.cyclic_lane_capacity = r.i32();
    fp.snapshot_version     = r.u32();
    const std::uint64_t text_n =
        static_cast<std::uint64_t>(fp.text_plane_count) * kDiskFingerprintPlanes;
    const std::uint64_t backend_n =
        static_cast<std::uint64_t>(fp.backend_plane_count) * kDiskFingerprintPlanes;
    if (text_n / kDiskFingerprintPlanes != fp.text_plane_count ||
        backend_n / kDiskFingerprintPlanes != fp.backend_plane_count ||
        r.remain() < text_n || r.remain() - text_n < backend_n) {
        throw std::runtime_error("KV disk FINGERPRINT plane list is truncated");
    }
    fp.text_plane_bytes.resize(static_cast<std::size_t>(text_n));
    r.raw(fp.text_plane_bytes.data(), fp.text_plane_bytes.size());
    fp.backend_plane_bytes.resize(static_cast<std::size_t>(backend_n));
    r.raw(fp.backend_plane_bytes.data(), fp.backend_plane_bytes.size());
    return fp;
}

void encode_slot(OutBuf& w, const DiskCheckpointSlot& slot) {
    w.u32(slot.frontier);
    w.u32(static_cast<std::uint32_t>(slot.kind));
    w.u64(slot.hash.lo);
    w.u64(slot.hash.hi);
    w.u64(slot.gdn_id);
    w.u64(slot.hidden_id);
    w.u64(slot.dflash_id);
}

DiskCheckpointSlot decode_slot(InBuf& r) {
    DiskCheckpointSlot slot;
    slot.frontier  = r.u32();
    slot.kind      = static_cast<ContextCheckpointKind>(r.u32());
    slot.hash.lo   = r.u64();
    slot.hash.hi   = r.u64();
    slot.gdn_id    = r.u64();
    slot.hidden_id = r.u64();
    slot.dflash_id = r.u64();
    return slot;
}

[[nodiscard]] std::vector<std::uint8_t> encode_meta(const DiskMeta& meta) {
    OutBuf w;
    w.raw(kDiskMetaMagic, 8);
    w.u32(kDiskFormatVersion);
    w.u64(meta.entry_id);
    w.u32(meta.execution_frontier);
    w.u32(meta.ledger_frontier);
    w.i32(meta.rope_delta);
    w.u32(meta.text_kv_valid);
    w.u32(meta.mtp_kv_valid);
    w.u32(meta.dflash_context_frontier);
    w.u8(meta.tail_hidden_valid ? 1 : 0);
    w.u8(meta.rewrite_valid ? 1 : 0);
    w.u8(static_cast<std::uint8_t>(meta.rewrite_kind));
    w.u8(meta.hash_c_valid ? 1 : 0);
    w.u32(meta.rewrite_frontier);
    w.u64(meta.hash_f.lo);
    w.u64(meta.hash_f.hi);
    w.u64(meta.hash_c.lo);
    w.u64(meta.hash_c.hi);
    w.u64(meta.ledger_id);
    w.u64(meta.identity_id);
    w.u64(meta.current_gdn_id);
    w.u64(meta.current_hidden_id);
    w.u64(meta.current_cyclic_id);
    w.u64(meta.rewrite_gdn_id);
    w.u64(meta.rewrite_hidden_id);
    w.u64(meta.rewrite_cyclic_id);
    encode_slot(w, meta.rollback);
    encode_slot(w, meta.ladders[0]);
    encode_slot(w, meta.ladders[1]);
    w.u32(static_cast<std::uint32_t>(meta.main_page_ids.size()));
    w.u32(static_cast<std::uint32_t>(meta.backend_page_ids.size()));
    for (std::uint64_t id : meta.main_page_ids) { w.u64(id); }
    for (std::uint64_t id : meta.backend_page_ids) { w.u64(id); }
    return std::move(w.bytes);
}

[[nodiscard]] std::optional<DiskMeta> try_decode_meta(const std::vector<std::uint8_t>& bytes) {
    try {
        if (bytes.size() < 16 || std::memcmp(bytes.data(), kDiskMetaMagic, 8) != 0) {
            return std::nullopt;
        }
        InBuf r{bytes.data() + 8, bytes.data() + bytes.size()};
        if (r.u32() != kDiskFormatVersion) { return std::nullopt; }
        DiskMeta meta;
        meta.entry_id                 = r.u64();
        meta.execution_frontier       = r.u32();
        meta.ledger_frontier          = r.u32();
        meta.rope_delta               = r.i32();
        meta.text_kv_valid            = r.u32();
        meta.mtp_kv_valid             = r.u32();
        meta.dflash_context_frontier  = r.u32();
        meta.tail_hidden_valid        = r.u8() != 0;
        meta.rewrite_valid            = r.u8() != 0;
        meta.rewrite_kind             = static_cast<RewriteCheckpointKind>(r.u8());
        meta.hash_c_valid             = r.u8() != 0;
        meta.rewrite_frontier         = r.u32();
        meta.hash_f.lo                = r.u64();
        meta.hash_f.hi                = r.u64();
        meta.hash_c.lo                = r.u64();
        meta.hash_c.hi                = r.u64();
        meta.ledger_id                = r.u64();
        meta.identity_id              = r.u64();
        meta.current_gdn_id           = r.u64();
        meta.current_hidden_id        = r.u64();
        meta.current_cyclic_id        = r.u64();
        meta.rewrite_gdn_id           = r.u64();
        meta.rewrite_hidden_id        = r.u64();
        meta.rewrite_cyclic_id        = r.u64();
        meta.rollback                 = decode_slot(r);
        meta.ladders[0]               = decode_slot(r);
        meta.ladders[1]               = decode_slot(r);
        if ((meta.rewrite_valid &&
             !valid_rewrite_kind(static_cast<std::uint8_t>(meta.rewrite_kind))) ||
            (meta.rollback.frontier != 0 &&
             (meta.rollback.kind != ContextCheckpointKind::TurnRollback ||
              !valid_checkpoint_kind(static_cast<std::uint32_t>(meta.rollback.kind)))) ||
            (meta.ladders[0].frontier != 0 &&
             (meta.ladders[0].kind != ContextCheckpointKind::Ladder ||
              !valid_checkpoint_kind(static_cast<std::uint32_t>(meta.ladders[0].kind)))) ||
            (meta.ladders[1].frontier != 0 &&
             (meta.ladders[1].kind != ContextCheckpointKind::Ladder ||
              !valid_checkpoint_kind(static_cast<std::uint32_t>(meta.ladders[1].kind))))) {
            return std::nullopt;
        }
        const auto main_n             = r.u32();
        const auto backend_n          = r.u32();
        if (main_n > kDiskMetaMaxPageIds || backend_n > kDiskMetaMaxPageIds) {
            return std::nullopt;
        }
        const std::uint64_t need =
            static_cast<std::uint64_t>(main_n) * 8ULL + static_cast<std::uint64_t>(backend_n) * 8ULL;
        if (need / 8ULL != static_cast<std::uint64_t>(main_n) + backend_n || r.remain() < need) {
            return std::nullopt;
        }
        meta.main_page_ids.resize(main_n);
        meta.backend_page_ids.resize(backend_n);
        for (std::uint32_t i = 0; i < main_n; ++i) { meta.main_page_ids[i] = r.u64(); }
        for (std::uint32_t i = 0; i < backend_n; ++i) { meta.backend_page_ids[i] = r.u64(); }
        return meta;
    } catch (...) { return std::nullopt; }
}

[[nodiscard]] std::uint64_t file_size_or_zero(const std::filesystem::path& path) {
    std::error_code ec;
    const auto n = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<std::uint64_t>(n);
}

class ReadOnlyMappedFile {
  public:
    explicit ReadOnlyMappedFile(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) { throw std::runtime_error("failed to read " + path.string()); }
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size < 0 ||
            static_cast<std::uint64_t>(st.st_size) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            const int saved = errno;
            ::close(fd_);
            fd_ = -1;
            errno = saved;
            throw std::runtime_error("KV disk map cannot be represented in host memory: " +
                                     path.string());
        }
        size_ = static_cast<std::size_t>(st.st_size);
        if (size_ != 0) {
            void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (mapped == MAP_FAILED) {
                ::close(fd_);
                fd_ = -1;
                throw std::runtime_error("failed to map " + path.string());
            }
            data_ = static_cast<const std::uint8_t*>(mapped);
        }
    }

    ReadOnlyMappedFile(const ReadOnlyMappedFile&) = delete;
    ReadOnlyMappedFile& operator=(const ReadOnlyMappedFile&) = delete;

    ~ReadOnlyMappedFile() {
        if (data_ != nullptr) { (void)::munmap(const_cast<std::uint8_t*>(data_), size_); }
        if (fd_ >= 0) { (void)::close(fd_); }
    }

    [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
        return {data_, size_};
    }

  private:
    int fd_ = -1;
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
};

struct ScopedFd {
    int value = -1;
    ScopedFd() = default;
    explicit ScopedFd(int fd) : value(fd) {}
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ~ScopedFd() { if (value >= 0) { (void)::close(value); } }
};

[[nodiscard]] std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path,
                                                         std::uint64_t max_bytes = kDiskHostFileMaxBytes) {
    const std::uint64_t sz = file_size_or_zero(path);
    if (sz > max_bytes) {
        throw std::runtime_error("KV disk file exceeds host read bound: " + path.string());
    }
    if (sz > std::numeric_limits<std::size_t>::max() ||
        sz > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("KV disk file cannot be represented in host memory: " +
                                 path.string());
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("failed to read " + path.string()); }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(sz));
    if (sz != 0) {
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(sz));
        if (!in) { throw std::runtime_error("failed to read " + path.string()); }
    }
    return bytes;
}

void write_file_bytes(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { throw std::runtime_error("failed to write " + path.string()); }
    if (bytes != 0) {
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    }
    if (!out) { throw std::runtime_error("failed to write " + path.string()); }
}

void compare_fingerprint_field(bool ok, const char* field) {
    if (!ok) {
        throw std::runtime_error(std::string("KV disk fingerprint mismatch: ") + field);
    }
}

void check_fingerprint(const DiskFingerprint& have, const DiskFingerprint& want) {
    compare_fingerprint_field(have.model_id == want.model_id, "model_id");
    compare_fingerprint_field(have.weights_id == want.weights_id, "weights_id");
    compare_fingerprint_field(have.kv_cache == want.kv_cache, "kv_cache");
    compare_fingerprint_field(have.speculative == want.speculative, "speculative");
    compare_fingerprint_field(have.page_size == want.page_size, "page_size");
    compare_fingerprint_field(have.text_plane_count == want.text_plane_count, "text_plane_count");
    compare_fingerprint_field(have.backend_plane_count == want.backend_plane_count,
                              "backend_plane_count");
    compare_fingerprint_field(have.gdn_conv_bytes == want.gdn_conv_bytes, "gdn_conv_bytes");
    compare_fingerprint_field(have.gdn_recurrent_bytes == want.gdn_recurrent_bytes,
                              "gdn_recurrent_bytes");
    compare_fingerprint_field(have.cyclic_lane_bytes == want.cyclic_lane_bytes, "cyclic_lane_bytes");
    compare_fingerprint_field(have.cyclic_layers == want.cyclic_layers, "cyclic_layers");
    compare_fingerprint_field(have.cyclic_capacity == want.cyclic_capacity, "cyclic_capacity");
    compare_fingerprint_field(have.cyclic_padded == want.cyclic_padded, "cyclic_padded");
    compare_fingerprint_field(have.cyclic_kv_heads == want.cyclic_kv_heads, "cyclic_kv_heads");
    compare_fingerprint_field(have.cyclic_head_dim == want.cyclic_head_dim, "cyclic_head_dim");
    compare_fingerprint_field(have.cyclic_lane_capacity == want.cyclic_lane_capacity,
                              "cyclic_lane_capacity");
    compare_fingerprint_field(have.snapshot_version == want.snapshot_version, "snapshot_version");
    compare_fingerprint_field(have.text_plane_bytes == want.text_plane_bytes, "text_plane_bytes");
    compare_fingerprint_field(have.backend_plane_bytes == want.backend_plane_bytes,
                              "backend_plane_bytes");
}

[[nodiscard]] PrefixReusePath rewrite_path(RewriteCheckpointKind kind) {
    return kind == RewriteCheckpointKind::ResponseReplay
               ? PrefixReusePath::RestoreResponseCheckpoint
               : PrefixReusePath::RestoreTurnCheckpoint;
}

[[nodiscard]] bool checkpoint_missing_hidden(std::uint64_t hidden_id,
                                              std::uint64_t hidden_bytes) noexcept {
    return hidden_bytes != 0 && hidden_id == 0;
}

[[nodiscard]] std::uint64_t expected_state_bytes(const DiskFingerprint& fp, DiskStateKind kind,
                                                 std::uint64_t hidden_bytes) {
    switch (kind) {
    case DiskStateKind::CurrentGdn:
    case DiskStateKind::RewriteGdn:
    case DiskStateKind::RollbackGdn:
    case DiskStateKind::LadderGdn:
        return fp.gdn_conv_bytes + fp.gdn_recurrent_bytes;
    case DiskStateKind::DflashLocal:
    case DiskStateKind::DflashRewrite:
    case DiskStateKind::DflashRollback:
    case DiskStateKind::DflashLadder:
        return fp.cyclic_lane_bytes;
    case DiskStateKind::TailHidden:
    case DiskStateKind::RewriteHidden:
    case DiskStateKind::RollbackHidden:
    case DiskStateKind::LadderHidden:
        return hidden_bytes;
    default:
        return 0;
    }
}

[[nodiscard]] bool owned_id_kind_conflict(
    const std::vector<std::pair<DiskObjectKind, std::uint64_t>>& owned) {
    for (std::size_t i = 0; i < owned.size(); ++i) {
        if (owned[i].second == 0) { continue; }
        for (std::size_t j = i + 1; j < owned.size(); ++j) {
            if (owned[j].second == owned[i].second && owned[i].first != owned[j].first) {
                return true;
            }
        }
    }
    return false;
}

struct BoundedIdSet {
    explicit BoundedIdSet(std::uint32_t n) : limit(n) {}
    std::uint32_t limit = 0;
    std::uint32_t seen  = 0;
    std::priority_queue<std::uint64_t> heap;

    void consider(std::uint64_t id) {
        ++seen;
        if (limit == 0) { return; }
        if (heap.size() < limit) {
            heap.push(id);
        } else if (id < heap.top()) {
            heap.pop();
            heap.push(id);
        }
    }

    [[nodiscard]] std::vector<std::uint64_t> take_sorted() {
        std::vector<std::uint64_t> ids;
        ids.reserve(heap.size());
        while (!heap.empty()) {
            ids.push_back(heap.top());
            heap.pop();
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }
};

void append_meta_objects(std::vector<std::pair<DiskObjectKind, std::uint64_t>>& out,
                         const DiskMeta& meta) {
    auto add = [&](DiskObjectKind kind, std::uint64_t id) {
        if (id != 0) { out.push_back({kind, id}); }
    };
    add(DiskObjectKind::Ledger, meta.ledger_id);
    add(DiskObjectKind::Identity, meta.identity_id);
    add(DiskObjectKind::State, meta.current_gdn_id);
    add(DiskObjectKind::State, meta.current_hidden_id);
    add(DiskObjectKind::State, meta.current_cyclic_id);
    add(DiskObjectKind::State, meta.rewrite_gdn_id);
    add(DiskObjectKind::State, meta.rewrite_hidden_id);
    add(DiskObjectKind::State, meta.rewrite_cyclic_id);
    add(DiskObjectKind::State, meta.rollback.gdn_id);
    add(DiskObjectKind::State, meta.rollback.hidden_id);
    add(DiskObjectKind::State, meta.rollback.dflash_id);
    for (const auto& slot : meta.ladders) {
        add(DiskObjectKind::State, slot.gdn_id);
        add(DiskObjectKind::State, slot.hidden_id);
        add(DiskObjectKind::State, slot.dflash_id);
    }
    for (std::uint64_t id : meta.main_page_ids) { add(DiskObjectKind::Main, id); }
    for (std::uint64_t id : meta.backend_page_ids) { add(DiskObjectKind::Backend, id); }
}

[[nodiscard]] std::vector<std::uint8_t> encode_tombstone(
    const std::vector<std::pair<DiskObjectKind, std::uint64_t>>& objects) {
    OutBuf w;
    w.raw(kDiskTombstoneMagic, 8);
    w.u32(kDiskFormatVersion);
    w.u32(static_cast<std::uint32_t>(objects.size()));
    for (const auto& [kind, id] : objects) {
        w.u8(static_cast<std::uint8_t>(kind));
        w.u64(id);
    }
    return std::move(w.bytes);
}

[[nodiscard]] std::optional<std::vector<std::pair<DiskObjectKind, std::uint64_t>>>
try_decode_tombstone(const std::vector<std::uint8_t>& bytes) {
    if (bytes.empty()) { return std::nullopt; }
    try {
        if (bytes.size() < 16 || std::memcmp(bytes.data(), kDiskTombstoneMagic, 8) != 0) {
            return std::nullopt;
        }
        InBuf r{bytes.data() + 8, bytes.data() + bytes.size()};
        if (r.u32() != kDiskFormatVersion) { return std::nullopt; }
        const auto n = r.u32();
        if (n > (1u << 20)) { return std::nullopt; }
        const std::uint64_t need = static_cast<std::uint64_t>(n) * 9ULL;
        if (need / 9ULL != n || r.remain() < need) { return std::nullopt; }
        std::vector<std::pair<DiskObjectKind, std::uint64_t>> objects;
        objects.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto kind = static_cast<DiskObjectKind>(r.u8());
            const auto id   = r.u64();
            if (static_cast<std::uint8_t>(kind) > static_cast<std::uint8_t>(DiskObjectKind::Identity)) {
                return std::nullopt;
            }
            if (id != 0) { objects.push_back({kind, id}); }
        }
        return objects;
    } catch (...) { return std::nullopt; }
}

} // namespace

void write_plane_fingerprint(std::vector<std::uint8_t>& out, const Tensor& plane,
                             PagedKVPlaneOrder order) {
    OutBuf w;
    append_plane_fingerprint(w, plane, order);
    out.insert(out.end(), w.bytes.begin(), w.bytes.end());
}

DiskFingerprint make_disk_fingerprint(std::string model_id, std::string weights_id,
                                      KvCacheStorage kv_cache, SpeculativeBackend speculative,
                                      const PagedKVPool& text, const PagedKVPool* backend,
                                      const LinearAttentionStatePool* gdn,
                                      const CyclicKVCache* cyclic) {
    DiskFingerprint fp;
    fp.model_id            = std::move(model_id);
    fp.weights_id          = std::move(weights_id);
    fp.kv_cache            = kv_cache;
    fp.speculative         = speculative;
    fp.page_size           = static_cast<std::uint32_t>(kPagedKVPageSize);
    fp.text_plane_count    = static_cast<std::uint32_t>(text.plane_count());
    fp.backend_plane_count = backend ? static_cast<std::uint32_t>(backend->plane_count()) : 0;
    for (std::uint32_t i = 0; i < fp.text_plane_count; ++i) {
        write_plane_fingerprint(fp.text_plane_bytes, text.plane(i), text.plane_order());
    }
    if (backend != nullptr) {
        for (std::uint32_t i = 0; i < fp.backend_plane_count; ++i) {
            write_plane_fingerprint(fp.backend_plane_bytes, backend->plane(i),
                                    backend->plane_order());
        }
    }
    if (gdn != nullptr) {
        fp.gdn_conv_bytes      = gdn->conv_host_image_bytes();
        fp.gdn_recurrent_bytes = gdn->recurrent_host_image_bytes();
    }
    if (cyclic != nullptr) {
        fp.cyclic_lane_bytes    = cyclic->lane_host_bytes();
        fp.cyclic_layers        = cyclic->layer_count();
        fp.cyclic_capacity      = cyclic->capacity();
        fp.cyclic_padded        = cyclic->padded_capacity();
        fp.cyclic_kv_heads      = cyclic->num_kv_heads();
        fp.cyclic_head_dim      = cyclic->head_dim();
        fp.cyclic_lane_capacity = cyclic->lane_capacity();
    }
    fp.snapshot_version = kDiskFormatVersion;
    return fp;
}

KVDiskCache::KVDiskCache(DiskOpenConfig config) : config_(std::move(config)) {
    test_fault_point_.store(config_.test_fault_point, std::memory_order_relaxed);
    if (config_.capacity_bytes == 0) {
        throw std::invalid_argument("KV disk cache requires a positive capacity");
    }
    if (config_.location.empty()) {
        throw std::invalid_argument("KV disk cache requires a location");
    }
    const std::size_t page_bytes =
        std::max<std::size_t>(config_.logical_page_bytes, 256);
    const std::uint32_t readers = std::min(
        16u, std::max(1u, config_.restore_io_threads == 0 ? 1u : config_.restore_io_threads));
    restore_io_threads_ = readers;
    std::uint32_t slots =
        config_.restore_window_slots == 0 ? 2u : config_.restore_window_slots;
    if (readers > 1) { slots = std::max(slots, 2u * readers); }
    slots = std::min(64u, std::max(2u, slots));
    window_.resize(slots);
    restore_window_stride_ = aligned_page_file_bytes(page_bytes);
    restore_window_bytes_ = restore_window_stride_ * static_cast<std::size_t>(slots);
    page_staging_stride_ = page_bytes;
    page_staging_bytes_ = page_staging_stride_ * static_cast<std::size_t>(slots);
    std::size_t max_state_object_bytes = std::max<std::size_t>(config_.gdn_staging_bytes, 256);
    max_state_object_bytes = std::max(
        max_state_object_bytes,
        static_cast<std::size_t>(config_.fingerprint.gdn_conv_bytes +
                                  config_.fingerprint.gdn_recurrent_bytes));
    max_state_object_bytes =
        std::max(max_state_object_bytes,
                 static_cast<std::size_t>(config_.fingerprint.cyclic_lane_bytes));
    max_state_object_bytes =
        std::max(max_state_object_bytes, static_cast<std::size_t>(config_.hidden_bytes));
    auto aligned_state_bytes = [](std::size_t bytes) {
        return (bytes + kDiskPageIoAlignment - 1) & ~(kDiskPageIoAlignment - 1);
    };
    const std::size_t conv_bytes =
        static_cast<std::size_t>(config_.fingerprint.gdn_conv_bytes);
    const std::size_t rec_bytes =
        static_cast<std::size_t>(config_.fingerprint.gdn_recurrent_bytes);
    const std::size_t hidden_bytes = static_cast<std::size_t>(config_.hidden_bytes);
    const std::size_t cyclic_bytes =
        static_cast<std::size_t>(config_.fingerprint.cyclic_lane_bytes);
    restore_state_arena_bytes_ = 2 * (aligned_state_bytes(conv_bytes) +
                                      aligned_state_bytes(rec_bytes) +
                                      aligned_state_bytes(hidden_bytes) +
                                      aligned_state_bytes(cyclic_bytes));
    spill_scratch_.resize(page_bytes + kDiskPageHeaderBytes);
    spill_zstd_.resize(std::max<std::size_t>(max_state_object_bytes * 2, 4096));
    decode_zstd_.resize(std::max<std::size_t>(max_state_object_bytes * 2, 4096));
    auto require_cuda = [](cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorName(err));
        }
    };
    try {
        require_cuda(cudaHostAlloc(&restore_window_allocation_,
                                   restore_window_bytes_ + kDiskPageIoAlignment - 1,
                                   cudaHostAllocDefault),
                     "KV disk restore window allocation failed");
        const auto raw = reinterpret_cast<std::uintptr_t>(restore_window_allocation_);
        restore_window_mem_ = reinterpret_cast<void*>(
            (raw + kDiskPageIoAlignment - 1) &
            ~(static_cast<std::uintptr_t>(kDiskPageIoAlignment) - 1));
        for (std::uint32_t i = 0; i < slots; ++i) {
            window_[i].io = static_cast<std::uint8_t*>(restore_window_mem_) +
                            restore_window_stride_ * static_cast<std::size_t>(i);
            window_[i].host = static_cast<std::uint8_t*>(window_[i].io) + kDiskPageHeaderBytes;
        }
        if (restore_state_arena_bytes_ != 0) {
            require_cuda(cudaHostAlloc(&restore_state_allocation_,
                                       restore_state_arena_bytes_ + kDiskPageIoAlignment - 1,
                                       cudaHostAllocDefault),
                         "KV disk immediate-state arena allocation failed");
            const auto state_raw = reinterpret_cast<std::uintptr_t>(restore_state_allocation_);
            restore_state_mem_ = reinterpret_cast<void*>(
                (state_raw + kDiskPageIoAlignment - 1) &
                ~(static_cast<std::uintptr_t>(kDiskPageIoAlignment) - 1));
            std::size_t state_offset = 0;
            auto take_state_slice = [&](std::size_t bytes) {
                auto* data = static_cast<std::uint8_t*>(restore_state_mem_) + state_offset;
                state_offset += aligned_state_bytes(bytes);
                return std::span<std::uint8_t>(data, bytes);
            };
            restore_state_capacity_.gdn_conv = take_state_slice(conv_bytes);
            restore_state_capacity_.gdn_rec = take_state_slice(rec_bytes);
            restore_state_capacity_.hidden = take_state_slice(hidden_bytes);
            restore_state_capacity_.cyclic = take_state_slice(cyclic_bytes);
            restore_state_capacity_.rewrite_gdn_conv = take_state_slice(conv_bytes);
            restore_state_capacity_.rewrite_gdn_rec = take_state_slice(rec_bytes);
            restore_state_capacity_.rewrite_hidden = take_state_slice(hidden_bytes);
            restore_state_capacity_.rewrite_cyclic = take_state_slice(cyclic_bytes);
        }
        require_cuda(cudaMalloc(&page_staging_, page_staging_bytes_),
                     "KV disk page device staging allocation failed");
        require_cuda(cudaStreamCreateWithFlags(&page_scatter_stream_, cudaStreamNonBlocking),
                     "KV disk page scatter stream failed");
        auto upload_scatter_plan = [&](const PagedKVPool* pool, PagedKVScatterPlane*& device,
                                       std::size_t& count, std::size_t& max_plane,
                                       std::size_t& logical_bytes) {
            if (pool == nullptr) { return; }
            PagedKVScatterPlan plan = make_paged_kv_scatter_plan(*pool);
            if (plan.page_bytes == 0 || plan.page_bytes > page_bytes || plan.planes.empty()) {
                throw std::logic_error("KV disk page scatter plan exceeds staging geometry");
            }
            count         = plan.planes.size();
            max_plane     = plan.max_plane_bytes;
            logical_bytes = plan.page_bytes;
            require_cuda(cudaMalloc(&device, count * sizeof(PagedKVScatterPlane)),
                         "KV disk page scatter descriptor allocation failed");
            require_cuda(cudaMemcpy(device, plan.planes.data(),
                                    count * sizeof(PagedKVScatterPlane), cudaMemcpyHostToDevice),
                         "KV disk page scatter descriptor upload failed");
        };
        upload_scatter_plan(config_.text_pool, text_scatter_planes_, text_scatter_plane_count_,
                            text_scatter_max_plane_bytes_, text_scatter_page_bytes_);
        upload_scatter_plan(config_.backend_pool, backend_scatter_planes_,
                            backend_scatter_plane_count_, backend_scatter_max_plane_bytes_,
                            backend_scatter_page_bytes_);
        require_cuda(cudaEventCreateWithFlags(&state_arena_idle_, cudaEventDisableTiming),
                     "KV disk immediate-state arena event failed");
        require_cuda(cudaStreamCreateWithFlags(&state_h2d_stream_, cudaStreamNonBlocking),
                     "KV disk state H2D stream failed");
        require_cuda(cudaEventCreateWithFlags(&copies_start_, cudaEventDefault),
                     "KV disk copies_start event failed");
        for (WindowSlot& slot : window_) {
            require_cuda(cudaEventCreateWithFlags(&slot.h2d_arrived, cudaEventDisableTiming),
                         "KV disk restore H2D-arrived event failed");
            require_cuda(cudaEventCreateWithFlags(&slot.h2d_event, cudaEventDisableTiming),
                         "KV disk restore H2D event failed");
        }
        open_directory();
        load_or_write_fingerprint();
        open_pack_store();
        load_index();
        validate_live_direct_segments();
        io_thread_ = std::thread([this] { io_loop(); });
        if (restore_io_threads_ > 1) {
            restore_threads_.reserve(restore_io_threads_);
            for (std::uint32_t i = 0; i < restore_io_threads_; ++i) {
                restore_threads_.emplace_back([this] { restore_loop(); });
            }
        }
    } catch (...) {
        stop_io_threads();
        if (copies_start_ != nullptr) { (void)cudaEventDestroy(copies_start_); }
        if (copies_done_ != nullptr) { (void)cudaEventDestroy(copies_done_); }
        if (state_arena_idle_ != nullptr) { (void)cudaEventDestroy(state_arena_idle_); }
        if (state_h2d_stream_ != nullptr) { (void)cudaStreamDestroy(state_h2d_stream_); }
        scatter_record_continue_.store(true, std::memory_order_release);
        if (scatter_test_gate_stream_ != nullptr) {
            (void)cudaStreamSynchronize(scatter_test_gate_stream_);
        }
        if (scatter_test_gate_ != nullptr) { (void)cudaEventDestroy(scatter_test_gate_); }
        if (scatter_test_gate_stream_ != nullptr) {
            (void)cudaStreamDestroy(scatter_test_gate_stream_);
        }
        if (page_scatter_stream_ != nullptr) { (void)cudaStreamDestroy(page_scatter_stream_); }
        for (RetiredCopyEvent& item : retired_copy_events_) {
            if (item.start != nullptr) { (void)cudaEventDestroy(item.start); }
            if (item.event != nullptr) { (void)cudaEventDestroy(item.event); }
            item.start = nullptr;
            item.event = nullptr;
        }
        retired_copy_events_.clear();
        for (WindowSlot& slot : window_) {
            if (slot.h2d_arrived != nullptr) { (void)cudaEventDestroy(slot.h2d_arrived); }
            if (slot.h2d_event != nullptr) { (void)cudaEventDestroy(slot.h2d_event); }
            slot.h2d_arrived = nullptr;
            slot.h2d_event = nullptr;
        }
        if (text_scatter_planes_ != nullptr) { (void)cudaFree(text_scatter_planes_); }
        if (backend_scatter_planes_ != nullptr) { (void)cudaFree(backend_scatter_planes_); }
        if (page_staging_ != nullptr) { (void)cudaFree(page_staging_); }
        if (restore_state_allocation_ != nullptr) {
            (void)cudaFreeHost(restore_state_allocation_);
        }
        if (restore_window_allocation_ != nullptr) {
            (void)cudaFreeHost(restore_window_allocation_);
        }
        copies_start_        = nullptr;
        copies_done_         = nullptr;
        state_arena_idle_    = nullptr;
        state_h2d_stream_    = nullptr;
        page_scatter_stream_ = nullptr;
        text_scatter_planes_ = nullptr;
        backend_scatter_planes_ = nullptr;
        page_staging_        = nullptr;
        restore_state_allocation_ = nullptr;
        restore_state_mem_   = nullptr;
        restore_window_allocation_ = nullptr;
        restore_window_mem_  = nullptr;
        release_lock();
        unregister_location();
        throw;
    }
}

KVDiskCache::~KVDiskCache() noexcept {
    try {
        stop_io_threads();
        try {
            queue_tombstone_object_unlinks();
            flush_queued_unlinks();
            write_manifest();
            fsync_store_dirs();
        } catch (...) { ++drops_; }
        release_lock();
        unregister_location();
        if (copies_start_ != nullptr) { (void)cudaEventDestroy(copies_start_); }
        if (copies_done_ != nullptr) { (void)cudaEventDestroy(copies_done_); }
        if (state_arena_h2d_pending_ && state_h2d_stream_ != nullptr) {
            (void)cudaStreamSynchronize(state_h2d_stream_);
        }
        if (state_arena_idle_ != nullptr) { (void)cudaEventDestroy(state_arena_idle_); }
        if (state_h2d_stream_ != nullptr) { (void)cudaStreamDestroy(state_h2d_stream_); }
        scatter_record_continue_.store(true, std::memory_order_release);
        if (scatter_test_gate_stream_ != nullptr) {
            (void)cudaStreamSynchronize(scatter_test_gate_stream_);
        }
        if (scatter_test_gate_ != nullptr) { (void)cudaEventDestroy(scatter_test_gate_); }
        if (scatter_test_gate_stream_ != nullptr) {
            (void)cudaStreamDestroy(scatter_test_gate_stream_);
        }
        if (page_scatter_stream_ != nullptr) { (void)cudaStreamDestroy(page_scatter_stream_); }
        for (RetiredCopyEvent& item : retired_copy_events_) {
            if (item.start != nullptr) { (void)cudaEventDestroy(item.start); }
            if (item.event != nullptr) { (void)cudaEventDestroy(item.event); }
            item.start = nullptr;
            item.event = nullptr;
        }
        retired_copy_events_.clear();
        for (WindowSlot& slot : window_) {
            if (slot.h2d_arrived != nullptr) { (void)cudaEventDestroy(slot.h2d_arrived); }
            if (slot.h2d_event != nullptr) { (void)cudaEventDestroy(slot.h2d_event); }
            slot.h2d_arrived = nullptr;
            slot.h2d_event = nullptr;
        }
        if (text_scatter_planes_ != nullptr) { (void)cudaFree(text_scatter_planes_); }
        if (backend_scatter_planes_ != nullptr) { (void)cudaFree(backend_scatter_planes_); }
        if (page_staging_ != nullptr) { (void)cudaFree(page_staging_); }
        if (restore_state_allocation_ != nullptr) {
            (void)cudaFreeHost(restore_state_allocation_);
        }
        if (restore_window_allocation_ != nullptr) {
            (void)cudaFreeHost(restore_window_allocation_);
        }
    } catch (...) {}
}

void KVDiskCache::register_location() {
    std::lock_guard lock(g_location_mu);
    const std::string key = canonical_location_.string();
    if (!g_locations.insert(key).second) {
        throw std::runtime_error("KV disk cache location is already open in this process");
    }
}

void KVDiskCache::unregister_location() noexcept {
    if (canonical_location_.empty()) { return; }
    std::lock_guard lock(g_location_mu);
    g_locations.erase(canonical_location_.string());
}

void KVDiskCache::open_directory() {
    std::error_code ec;
    if (std::filesystem::exists(config_.location, ec) &&
        !std::filesystem::is_directory(config_.location, ec)) {
        throw std::runtime_error("KV disk location exists and is not a directory");
    }
    std::filesystem::create_directories(config_.location);
    canonical_location_ = std::filesystem::weakly_canonical(config_.location);
    register_location();
    acquire_lock();
    std::filesystem::create_directories(config_.location / "packs");
    std::filesystem::create_directories(config_.location / "maps");
    // Keep empty legacy namespaces so stale external directory scans are
    // harmless. v4 never writes or reads object payloads through them.
    for (DiskObjectKind kind : {DiskObjectKind::Main, DiskObjectKind::Backend,
                                DiskObjectKind::State, DiskObjectKind::Ledger,
                                DiskObjectKind::Identity}) {
        std::filesystem::create_directories(config_.location / "objects" / kind_dir(kind));
    }
    std::filesystem::create_directories(config_.location / "entries");
    std::filesystem::create_directories(config_.location / "tmp");
    std::filesystem::create_directories(config_.location / "tombstones");
    std::error_code tec;
    for (const auto& entry : std::filesystem::directory_iterator(config_.location / "tmp", tec)) {
        std::filesystem::remove_all(entry.path(), tec);
    }
    fsync_store_dirs();
    const auto parent = config_.location.parent_path();
    if (!parent.empty()) { fsync_dir(parent); }
}

void KVDiskCache::acquire_lock() {
    const auto path = config_.location / "LOCK";
    lock_fd_        = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (lock_fd_ < 0) {
        unregister_location();
        throw std::runtime_error("failed to open KV disk LOCK");
    }
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        ::close(lock_fd_);
        lock_fd_ = -1;
        unregister_location();
        throw std::runtime_error("KV disk cache location is already locked");
    }
}

void KVDiskCache::release_lock() noexcept {
    if (lock_fd_ >= 0) {
        (void)::flock(lock_fd_, LOCK_UN);
        (void)::close(lock_fd_);
        lock_fd_ = -1;
    }
}

void KVDiskCache::fsync_path(const std::filesystem::path& path) const {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { throw std::runtime_error("failed to fsync " + path.string()); }
    if (::fsync(fd) != 0) {
        ::close(fd);
        throw std::runtime_error("fsync failed for " + path.string());
    }
    ::close(fd);
}

void KVDiskCache::fsync_new_objects(const SpillSession& session) const {
    std::unordered_set<std::string> touched;
    for (std::size_t i = 0; i < session.new_object_ids.size(); ++i) {
        const std::uint64_t id = session.new_object_ids[i];
        if (id == 0) { continue; }
        const auto it = objects_.find(id);
        if (it == objects_.end()) { throw std::runtime_error("KV disk new pack object is missing"); }
        touched.insert(pack_path(it->second.kind, it->second.location.segment).string());
    }
    for (const auto& path : touched) { fsync_path(path); }
    // A map Put is allowed to outlive its entry metadata, but it must never
    // outlive the segment pathname it names.  In particular, a newly rolled
    // segment's directory entry has to be durable before the map log can make
    // any of its extents discoverable after a crash.
    fsync_object_namespaces();
    maybe_test_fault(DiskFaultPoint::AfterPackSync);
    append_pack_map(session);
}

void KVDiskCache::fsync_dir(const std::filesystem::path& dir) const {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) { throw std::runtime_error("failed to fsync directory " + dir.string()); }
    if (::fsync(fd) != 0) {
        ::close(fd);
        throw std::runtime_error("directory fsync failed for " + dir.string());
    }
    ::close(fd);
}

void KVDiskCache::fsync_object_namespaces() const {
    const auto generation = config_.location / "packs" / std::to_string(pack_generation_);
    std::error_code ec;
    if (!std::filesystem::exists(generation, ec)) {
        fsync_dir(config_.location / "packs");
        fsync_dir(config_.location / "maps");
        return;
    }
    fsync_dir(generation / "main");
    fsync_dir(generation / "backend");
    fsync_dir(generation / "state");
    fsync_dir(generation / "ledger");
    fsync_dir(generation / "identity");
    fsync_dir(generation);
    fsync_dir(config_.location / "packs");
    fsync_dir(config_.location / "maps");
}

void KVDiskCache::fsync_store_dirs() const {
    fsync_entered_.store(true, std::memory_order_release);
    const int stall_ms = fsync_stall_ms_.load(std::memory_order_acquire);
    if (stall_ms > 0) { std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms)); }
    fsync_object_namespaces();
    fsync_dir(config_.location / "entries");
    fsync_dir(config_.location / "tmp");
    fsync_dir(config_.location / "tombstones");
    fsync_dir(config_.location);
}

void KVDiskCache::persist_eviction(std::unique_lock<std::mutex>& lock) {
    try {
        write_manifest(lock);
    } catch (...) { ++drops_; }
    if (lock.owns_lock()) { lock.unlock(); }
    try {
        fsync_store_dirs();
    } catch (...) { ++drops_; }
    if (!lock.owns_lock()) { lock.lock(); }
    reclaim_durable_tombstones();
}


std::filesystem::path KVDiskCache::entry_dir(std::uint64_t id) const {
    return config_.location / "entries" / std::to_string(id);
}

void KVDiskCache::write_fingerprint_file() const {
    const auto bytes = encode_fingerprint(config_.fingerprint);
    const auto path  = config_.location / "FINGERPRINT";
    write_file_bytes(path, bytes.data(), bytes.size());
    fsync_path(path);
}

void KVDiskCache::load_or_write_fingerprint() {
    const auto path = config_.location / "FINGERPRINT";
    std::error_code ec;
    const bool has_fp      = std::filesystem::exists(path, ec);
    const bool has_entries = std::filesystem::exists(config_.location / "entries", ec) &&
                             !std::filesystem::is_empty(config_.location / "entries", ec);
    const bool has_objects = std::filesystem::exists(config_.location / "objects", ec) ||
                             std::filesystem::exists(config_.location / "packs", ec);
    bool objects_nonempty  = false;
    if (has_objects) {
        const auto has_file = [&](const std::filesystem::path& dir) {
            std::error_code dir_ec;
            if (!std::filesystem::exists(dir, dir_ec)) { return false; }
            for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, dir_ec)) {
                if (entry.is_regular_file()) { return true; }
            }
            return false;
        };
        objects_nonempty = has_file(config_.location / "objects") ||
                           has_file(config_.location / "packs") ||
                           has_file(config_.location / "maps") ||
                           std::filesystem::exists(config_.location / "PACKSET", ec);
    }
    if (!has_fp && (has_entries || objects_nonempty)) {
        throw std::runtime_error("KV disk FINGERPRINT is missing from a non-empty tree");
    }
    if (!has_fp) {
        write_fingerprint_file();
        write_manifest();
        fsync_dir(config_.location);
        return;
    }
    check_fingerprint(decode_fingerprint(read_file_bytes(path)), config_.fingerprint);
}

std::filesystem::path KVDiskCache::pack_path(const PackGeneration& generation,
                                              DiskObjectKind kind,
                                              std::uint32_t segment) const {
    char name[32]{};
    std::snprintf(name, sizeof(name), "%08u.pack", segment);
    return generation.root / kind_dir(kind) / name;
}

std::filesystem::path KVDiskCache::pack_path(DiskObjectKind kind,
                                              std::uint32_t segment) const {
    if (!active_generation_) {
        throw std::logic_error("KV disk active pack generation is unavailable");
    }
    return pack_path(*active_generation_, kind, segment);
}

void KVDiskCache::persist_packset(std::uint64_t reserved_next_id) const {
    const auto tmp = config_.location / "tmp" / "PACKSET";
    const auto bytes = encode_packset(pack_generation_, reserved_next_id);
    write_file_bytes(tmp, bytes.data(), bytes.size());
    maybe_test_fault(DiskFaultPoint::AfterPacksetTmpWrite);
    fsync_path(tmp);
    maybe_test_fault(DiskFaultPoint::AfterPacksetTmpSync);
    std::filesystem::rename(tmp, config_.location / "PACKSET");
    maybe_test_fault(DiskFaultPoint::AfterPacksetRename);
    fsync_dir(config_.location);
    maybe_test_fault(DiskFaultPoint::AfterPacksetRootSync);
}

KVDiskCache::PackDescriptor::~PackDescriptor() {
    if (direct >= 0) { (void)::close(direct); }
    if (buffered >= 0) { (void)::close(buffered); }
}

std::shared_ptr<KVDiskCache::PackDescriptor> KVDiskCache::acquire_pack_descriptor(
    const std::shared_ptr<PackGeneration>& generation, DiskObjectKind kind,
    std::uint32_t segment, bool require_direct) const {
    if (!generation || generation->number == 0) {
        throw std::logic_error("KV disk pack generation lease is empty");
    }
    const auto path = pack_path(*generation, kind, segment);
    const std::string key = path.string();
    std::lock_guard guard(generation->descriptor_mu);
    auto& descriptor = generation->descriptors[key];
    if (!descriptor) {
        auto opened = std::make_shared<PackDescriptor>();
        opened->buffered = ::open(path.c_str(), O_RDWR);
        if (opened->buffered < 0) {
            throw std::runtime_error("failed to open KV disk pack segment");
        }
        // A direct descriptor is deliberately mandatory for page/raw-state
        // transfers.  Falling back here would silently invalidate the pinned
        // restore-slot throughput contract.
        opened->direct = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (opened->direct < 0 && require_direct) {
            throw std::runtime_error("failed to open direct KV disk pack segment");
        }
        descriptor = std::move(opened);
    }
    if (require_direct && descriptor->direct < 0) {
        throw std::runtime_error("direct KV disk pack descriptor is unavailable");
    }
    return descriptor;
}

void KVDiskCache::open_pack_store() {
    const auto packset = config_.location / "PACKSET";
    std::error_code ec;
    if (!std::filesystem::exists(packset, ec)) {
        const auto bootstrap_tree_is_nonempty = [&] {
            std::error_code tree_ec;
            const auto entries = config_.location / "entries";
            for (const auto& item : std::filesystem::directory_iterator(entries, tree_ec)) {
                if (tree_ec || item.is_directory(tree_ec) || item.is_regular_file(tree_ec)) {
                    return true;
                }
            }
            if (tree_ec) { return true; }
            const auto tombstones = config_.location / "tombstones";
            for (const auto& item : std::filesystem::directory_iterator(tombstones, tree_ec)) {
                if (tree_ec || item.is_directory(tree_ec) || item.is_regular_file(tree_ec)) {
                    return true;
                }
            }
            if (tree_ec) { return true; }
            const auto legacy = config_.location / "objects";
            for (const auto& item : std::filesystem::recursive_directory_iterator(legacy, tree_ec)) {
                if (tree_ec || item.is_regular_file(tree_ec)) { return true; }
            }
            if (tree_ec) { return true; }
            const auto maps = config_.location / "maps";
            const auto base_name = std::string("objects-1.base");
            const auto log_name = std::string("objects-1.log");
            for (const auto& item : std::filesystem::directory_iterator(maps, tree_ec)) {
                if (tree_ec) { return true; }
                if (!item.is_regular_file(tree_ec)) {
                    return true;
                }
                const auto name = item.path().filename();
                if (name != base_name && name != log_name) { return true; }
                const auto bytes = item.file_size(tree_ec);
                if (tree_ec || (name == base_name &&
                                bytes != static_cast<std::uintmax_t>(kPackMapBaseHeaderBytes + 4)) ||
                    (name == log_name && bytes != 0)) {
                    return true;
                }
            }
            if (tree_ec) { return true; }
            const auto packs = config_.location / "packs";
            for (const auto& generation : std::filesystem::directory_iterator(packs, tree_ec)) {
                if (tree_ec || !generation.is_directory(tree_ec) ||
                    generation.path().filename() != "1") {
                    return true;
                }
                for (const auto& kind_dir :
                     std::filesystem::directory_iterator(generation.path(), tree_ec)) {
                    if (tree_ec || !kind_dir.is_directory(tree_ec)) { return true; }
                    for (const auto& segment :
                         std::filesystem::directory_iterator(kind_dir.path(), tree_ec)) {
                        if (tree_ec || !segment.is_regular_file(tree_ec) ||
                            segment.path().filename() != "00000000.pack" ||
                            segment.file_size(tree_ec) != 0) {
                            return true;
                        }
                    }
                    if (tree_ec) { return true; }
                }
                if (tree_ec) { return true; }
            }
            return static_cast<bool>(tree_ec);
        };
        // A no-PACKSET tree is recoverable only when it contains the empty
        // bootstrap shape (or its own zero-length partial files).  Payloads,
        // entries, tombstones, legacy objects, and foreign generations are
        // never overwritten.
        if (!entries_.empty() || bootstrap_tree_is_nonempty()) {
            throw std::runtime_error("KV disk PACKSET is missing from a non-empty tree");
        }
        pack_generation_ = 1;
        next_object_id_ = 1;
        object_id_reservation_limit_ = (1ULL << 16) + 1;
        active_generation_ = std::make_shared<PackGeneration>();
        active_generation_->number = pack_generation_;
        active_generation_->root =
            config_.location / "packs" / std::to_string(pack_generation_);
        const auto base = config_.location / "maps" / "objects-1.base";
        const auto log = config_.location / "maps" / "objects-1.log";
        const auto empty_base = encode_pack_map_base(pack_generation_, next_object_id_, {});
        const auto publish_bootstrap_file = [&](const std::filesystem::path& path,
                                                const std::vector<std::uint8_t>& bytes) {
            const auto tmp = config_.location / "tmp" /
                             ("bootstrap-" + path.filename().string());
            write_file_bytes(tmp, bytes.data(), bytes.size());
            fsync_path(tmp);
            std::filesystem::rename(tmp, path);
            fsync_dir(path.parent_path());
        };
        publish_bootstrap_file(base, empty_base);
        publish_bootstrap_file(log, {});
        for (DiskObjectKind kind : {DiskObjectKind::Main, DiskObjectKind::Backend,
                                    DiskObjectKind::State, DiskObjectKind::Ledger,
                                    DiskObjectKind::Identity}) {
            std::filesystem::create_directories(pack_path(kind, 0).parent_path());
            const int fd = ::open(pack_path(kind, 0).c_str(), O_RDWR | O_CREAT, 0644);
            if (fd < 0) { throw std::runtime_error("failed to create KV disk pack segment"); }
            struct stat segment_stat {};
            if (::fstat(fd, &segment_stat) != 0 || segment_stat.st_size != 0) {
                ::close(fd);
                throw std::runtime_error("KV disk bootstrap pack segment is not empty");
            }
            if (::fsync(fd) != 0) { ::close(fd); throw std::runtime_error("failed to sync KV disk pack segment"); }
            ::close(fd);
            fsync_dir(pack_path(kind, 0).parent_path());
        }
        const auto tmp = config_.location / "tmp" / "PACKSET";
        const auto bytes = encode_packset(pack_generation_, object_id_reservation_limit_);
        write_file_bytes(tmp, bytes.data(), bytes.size());
        maybe_test_fault(DiskFaultPoint::AfterPacksetTmpWrite);
        fsync_path(tmp);
        maybe_test_fault(DiskFaultPoint::AfterPacksetTmpSync);
        std::filesystem::rename(tmp, packset);
        maybe_test_fault(DiskFaultPoint::AfterPacksetRename);
        fsync_dir(config_.location);
        maybe_test_fault(DiskFaultPoint::AfterPacksetRootSync);
    } else {
        const auto bytes = read_file_bytes(packset);
        if (!decode_packset(bytes, pack_generation_, next_object_id_)) {
            throw std::runtime_error("KV disk PACKSET is corrupt");
        }
        active_generation_ = std::make_shared<PackGeneration>();
        active_generation_->number = pack_generation_;
        active_generation_->root =
            config_.location / "packs" / std::to_string(pack_generation_);
        object_id_reservation_limit_ = next_object_id_;
    }
    load_pack_map();
    const auto remove_stale_pack_state = [&] {
        const std::string active = std::to_string(pack_generation_);
        bool packs_changed = false;
        bool maps_changed = false;
        std::error_code cleanup_ec;
        const auto packs = config_.location / "packs";
        for (const auto& item : std::filesystem::directory_iterator(packs, cleanup_ec)) {
            if (cleanup_ec) { throw std::runtime_error("failed to inspect KV pack generations"); }
            if (!item.is_directory(cleanup_ec)) {
                if (cleanup_ec) { throw std::runtime_error("failed to inspect KV pack generation"); }
                continue;
            }
            if (item.path().filename() == active) { continue; }
            std::uint64_t generation = 0;
            try {
                const std::string name = item.path().filename().string();
                std::size_t parsed = 0;
                generation = std::stoull(name, &parsed);
                if (parsed != name.size() || generation == 0) { continue; }
            } catch (...) {
                continue;
            }
            (void)generation;
            std::filesystem::remove_all(item.path(), cleanup_ec);
            if (cleanup_ec) { throw std::runtime_error("failed to remove stale KV pack generation"); }
            packs_changed = true;
        }
        const auto maps = config_.location / "maps";
        for (const auto& item : std::filesystem::directory_iterator(maps, cleanup_ec)) {
            if (cleanup_ec) { throw std::runtime_error("failed to inspect KV object maps"); }
            if (!item.is_regular_file(cleanup_ec)) {
                if (cleanup_ec) { throw std::runtime_error("failed to inspect KV object map"); }
                continue;
            }
            const std::string name = item.path().filename().string();
            constexpr std::string_view prefix = "objects-";
            constexpr std::string_view base_suffix = ".base";
            constexpr std::string_view log_suffix = ".log";
            const bool known_suffix = name.size() > prefix.size() + base_suffix.size() &&
                                      (name.ends_with(base_suffix) || name.ends_with(log_suffix));
            if (!name.starts_with(prefix) || !known_suffix) { continue; }
            const std::size_t end = name.rfind('.');
            const std::string number = name.substr(prefix.size(), end - prefix.size());
            try {
                std::size_t parsed = 0;
                const auto generation = std::stoull(number, &parsed);
                if (parsed != number.size() || generation == 0 || number == active) { continue; }
            } catch (...) {
                continue;
            }
            std::filesystem::remove(item.path(), cleanup_ec);
            if (cleanup_ec) { throw std::runtime_error("failed to remove stale KV object map"); }
            maps_changed = true;
        }
        if (packs_changed) { fsync_dir(packs); }
        if (maps_changed) { fsync_dir(maps); }
    };
    remove_stale_pack_state();
    object_id_reservation_limit_ = std::max(object_id_reservation_limit_, next_object_id_);
}

void KVDiskCache::validate_live_direct_segments() const {
    // Entry recovery has now assigned live/skip ownership and discarded
    // missing or corrupt entries.  Qualify every segment that can actually be
    // restored, while deliberately ignoring crash-left map Puts that were
    // never published by entry metadata.
    std::unordered_set<std::uint64_t> direct_segments;
    for (const auto& [id, ref] : objects_) {
        (void)id;
        if (ref.live_refs == 0 && ref.skip_refs == 0 && ref.hold_refs == 0) { continue; }
        if (ref.kind != DiskObjectKind::Main && ref.kind != DiskObjectKind::Backend &&
            ref.kind != DiskObjectKind::State) {
            continue;
        }
        const std::uint64_t key = (static_cast<std::uint64_t>(kind_index(ref.kind)) << 32) |
                                  ref.location.segment;
        if (direct_segments.insert(key).second) {
            const auto path = pack_path(*ref.location.generation, ref.kind, ref.location.segment);
            ScopedFd descriptor(::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC));
            if (descriptor.value < 0) {
                throw std::runtime_error("failed to open direct KV disk pack segment");
            }
        }
    }
}

void KVDiskCache::load_pack_map() {
    const auto base = config_.location / "maps" /
                      ("objects-" + std::to_string(pack_generation_) + ".base");
    const auto log = config_.location / "maps" /
                     ("objects-" + std::to_string(pack_generation_) + ".log");
    std::error_code ec;
    if (!std::filesystem::exists(base, ec) || !std::filesystem::exists(log, ec)) {
        throw std::runtime_error("KV disk object map is missing");
    }
    // Object-map size follows the configured cache population.  It is not a
    // small host-side metadata object and must not inherit the 64 MiB bound
    // used for fingerprints, entry metadata, ledgers, and identities.
    const ReadOnlyMappedFile base_file(base);
    const auto base_bytes = base_file.bytes();
    if (base_bytes.size() < static_cast<std::size_t>(kPackMapBaseHeaderBytes) + 4 ||
        std::memcmp(base_bytes.data(), kDiskObjectMapMagic, 8) != 0) {
        throw std::runtime_error("KV disk object-map base is corrupt");
    }
    if (crc32c(std::span<const std::uint8_t>(base_bytes.data(), base_bytes.size() - 4)) !=
        (static_cast<std::uint32_t>(base_bytes[base_bytes.size() - 4]) |
         (static_cast<std::uint32_t>(base_bytes[base_bytes.size() - 3]) << 8) |
         (static_cast<std::uint32_t>(base_bytes[base_bytes.size() - 2]) << 16) |
         (static_cast<std::uint32_t>(base_bytes[base_bytes.size() - 1]) << 24))) {
        throw std::runtime_error("KV disk object-map base CRC mismatch");
    }
    std::array<std::unordered_map<std::uint32_t,
                                  std::vector<std::pair<std::uint64_t, std::uint64_t>>>, 5>
        ranges;
    auto install = [this, &ranges](const PackedMapEntry& entry) {
        if (entry.id == 0 || kind_index(entry.kind) >= 5 || entry.extent_bytes == 0 ||
            entry.extent_bytes % kDiskPageIoAlignment != 0 || entry.stored_bytes == 0 ||
            entry.stored_bytes > entry.extent_bytes || entry.logical_bytes == 0 ||
            entry.offset % kDiskPageIoAlignment != 0 ||
            entry.extent_bytes > kPackSegmentBytes ||
            entry.offset > kPackSegmentBytes - entry.extent_bytes) {
            throw std::runtime_error("KV disk object-map location is invalid: id=" +
                                     std::to_string(entry.id) + " kind=" +
                                     std::to_string(static_cast<unsigned>(entry.kind)) +
                                     " extent=" + std::to_string(entry.extent_bytes) +
                                     " stored=" + std::to_string(entry.stored_bytes) +
                                     " logical=" + std::to_string(entry.logical_bytes));
        }
        ObjectRef ref;
        ref.kind = entry.kind; ref.bytes = entry.extent_bytes;
        ref.location.generation = active_generation_;
        ref.location.segment = entry.segment; ref.location.offset = entry.offset;
        ref.location.extent_bytes = entry.extent_bytes;
        ref.location.stored_bytes = entry.stored_bytes;
        ref.location.logical_bytes = entry.logical_bytes;
        ref.location.record_crc32c = entry.record_crc32c;
        const auto [it, inserted] = objects_.emplace(entry.id, ref);
        if (!inserted && (it->second.kind != ref.kind || it->second.bytes != ref.bytes ||
                          it->second.location.segment != ref.location.segment ||
                          it->second.location.offset != ref.location.offset ||
                          it->second.location.extent_bytes != ref.location.extent_bytes ||
                          it->second.location.stored_bytes != ref.location.stored_bytes ||
                          it->second.location.logical_bytes != ref.location.logical_bytes ||
                          it->second.location.record_crc32c != ref.location.record_crc32c)) {
            throw std::runtime_error("KV disk object-map id conflict");
        }
        if (!inserted) { return; }
        ranges[kind_index(entry.kind)][entry.segment].emplace_back(
            entry.offset, entry.offset + entry.extent_bytes);
        bump_next_id(next_object_id_, entry.id);
        const std::size_t index = kind_index(entry.kind);
        if (entry.segment > pack_active_segment_[index] ||
            (entry.segment == pack_active_segment_[index] &&
             entry.offset + entry.extent_bytes > pack_active_tail_[index])) {
            pack_active_segment_[index] = entry.segment;
            pack_active_tail_[index] = entry.offset + entry.extent_bytes;
        }
    };
    try {
        InBuf in{base_bytes.data() + 8, base_bytes.data() + base_bytes.size() - 4};
        if (in.u32() != kDiskFormatVersion || in.u32() != kPackMapBaseEndianLittle ||
            in.u64() != pack_generation_) {
            throw std::runtime_error("KV disk object-map base version is invalid");
        }
        bump_next_id(next_object_id_, in.u64());
        const std::uint64_t count = in.u64();
        const std::uint64_t payload_bytes = in.u64();
        if (count > std::numeric_limits<std::uint64_t>::max() / kPackMapPutBytes ||
            payload_bytes != count * kPackMapPutBytes || in.remain() != payload_bytes) {
            throw std::runtime_error("KV disk object-map base length is invalid");
        }
        for (std::uint64_t n = 0; n < count; ++n) {
            PackedMapEntry entry;
            entry.id = in.u64(); entry.kind = static_cast<DiskObjectKind>(in.u8()); in.skip(3);
            entry.segment = in.u32(); entry.offset = in.u64(); entry.extent_bytes = in.u64();
            entry.stored_bytes = in.u64(); entry.logical_bytes = in.u64();
            entry.record_crc32c = in.u32(); install(entry);
        }
    } catch (const std::runtime_error&) { throw; }
    std::size_t off = 0;
    std::size_t log_size = 0;
    {
        const ReadOnlyMappedFile log_file(log);
        const auto bytes = log_file.bytes();
        log_size = bytes.size();
        while (off < bytes.size()) {
            if (bytes.size() - off < static_cast<std::size_t>(kPackMapFrameBytes) + 4) { break; }
            const std::uint32_t frame = static_cast<std::uint32_t>(bytes[off]) |
                                        (static_cast<std::uint32_t>(bytes[off + 1]) << 8) |
                                        (static_cast<std::uint32_t>(bytes[off + 2]) << 16) |
                                        (static_cast<std::uint32_t>(bytes[off + 3]) << 24);
            if (frame != kPackMapFrameBytes) {
                throw std::runtime_error("KV disk object-map frame is corrupt");
            }
            const auto stored_crc = static_cast<std::uint32_t>(bytes[off + frame]) |
                                    (static_cast<std::uint32_t>(bytes[off + frame + 1]) << 8) |
                                    (static_cast<std::uint32_t>(bytes[off + frame + 2]) << 16) |
                                    (static_cast<std::uint32_t>(bytes[off + frame + 3]) << 24);
            if (crc32c(bytes.subspan(off, frame)) != stored_crc) {
                throw std::runtime_error("KV disk object-map CRC mismatch");
            }
            try {
                InBuf in{bytes.data() + off + 4, bytes.data() + off + frame};
                if (in.u8() != kPackMapTypePut || in.u8() != 0 || in.u8() != 0 || in.u8() != 0) {
                    throw std::runtime_error("KV disk object-map frame type is invalid");
                }
                PackedMapEntry entry;
                entry.id = in.u64();
                entry.kind = static_cast<DiskObjectKind>(in.u8());
                in.skip(3);
                entry.segment = in.u32();
                entry.offset = in.u64();
                entry.extent_bytes = in.u64();
                entry.stored_bytes = in.u64();
                entry.logical_bytes = in.u64();
                entry.record_crc32c = in.u32();
                install(entry);
            } catch (const std::runtime_error&) { throw; }
            off += static_cast<std::size_t>(frame) + 4;
        }
    }
    if (off != log_size) {
        const int fd = ::open(log.c_str(), O_RDWR);
        if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(off)) != 0 || ::fsync(fd) != 0) {
            if (fd >= 0) { ::close(fd); }
            throw std::runtime_error("failed to truncate KV disk object-map tail");
        }
        ::close(fd);
    }
    for (auto& by_segment : ranges) {
        for (auto& [segment, extents] : by_segment) {
            (void)segment;
            std::sort(extents.begin(), extents.end());
            for (std::size_t i = 1; i < extents.size(); ++i) {
                if (extents[i].first < extents[i - 1].second) {
                    throw std::runtime_error("KV disk object-map extents overlap");
                }
            }
        }
    }

    // The append frontier is a property of the durable pack namespace, not
    // merely of the append-only map.  A crash may leave an unpublished Put,
    // and older writers could also have made that Put durable before the
    // pathname of a newly rolled segment.  Such an unreachable Put must not
    // steer later appends to a segment that does not exist.  For an existing
    // segment retain both its physical tail and every mapped extent so that
    // partial/unpublished bytes are never overwritten.
    pack_active_segment_.fill(0);
    pack_active_tail_.fill(0);
    for (DiskObjectKind kind : {DiskObjectKind::Main, DiskObjectKind::Backend,
                                DiskObjectKind::State, DiskObjectKind::Ledger,
                                DiskObjectKind::Identity}) {
        const std::size_t index = kind_index(kind);
        const auto directory = active_generation_->root / kind_dir(kind);
        std::error_code scan_ec;
        for (const auto& item : std::filesystem::directory_iterator(directory, scan_ec)) {
            if (scan_ec || !item.is_regular_file(scan_ec)) {
                if (scan_ec) { throw std::runtime_error("failed to scan KV disk pack namespace"); }
                continue;
            }
            const std::string name = item.path().filename().string();
            if (name.size() != 13 || name.substr(8) != ".pack" ||
                !std::all_of(name.begin(), name.begin() + 8,
                             [](unsigned char c) { return c >= '0' && c <= '9'; })) {
                continue;
            }
            std::uint64_t parsed = 0;
            const auto [end, parse_ec] = std::from_chars(name.data(), name.data() + 8, parsed);
            if (parse_ec != std::errc{} || end != name.data() + 8 ||
                parsed > std::numeric_limits<std::uint32_t>::max()) {
                continue;
            }
            const std::uint32_t segment = static_cast<std::uint32_t>(parsed);
            const std::uint64_t file_bytes = item.file_size(scan_ec);
            if (scan_ec) { throw std::runtime_error("KV disk pack segment size is invalid"); }
            // Only mapped extents are semantic.  A sparse/corrupt suffix must
            // not make startup scale with, or reject, the apparent host file
            // size.  Mark an oversized segment full so a future append rolls.
            std::uint64_t tail = kPackSegmentBytes;
            if (file_bytes < kPackSegmentBytes) {
                tail = (file_bytes + kDiskPageIoAlignment - 1) &
                       ~(static_cast<std::uint64_t>(kDiskPageIoAlignment) - 1);
            }
            const auto range_it = ranges[index].find(segment);
            if (range_it != ranges[index].end()) {
                for (const auto& extent : range_it->second) { tail = std::max(tail, extent.second); }
            }
            if (segment > pack_active_segment_[index] ||
                (segment == pack_active_segment_[index] && tail > pack_active_tail_[index])) {
                pack_active_segment_[index] = segment;
                pack_active_tail_[index] = tail;
            }
        }
        if (scan_ec) { throw std::runtime_error("failed to scan KV disk pack namespace"); }
    }
}

void KVDiskCache::append_pack_map(const SpillSession& session) const {
    const auto log = config_.location / "maps" /
                     ("objects-" + std::to_string(pack_generation_) + ".log");
    const int fd = ::open(log.c_str(), O_WRONLY | O_APPEND);
    if (fd < 0) { throw std::runtime_error("failed to open KV disk object-map log"); }
    auto close_fd = [&] { ::close(fd); };
    for (std::uint64_t id : session.new_object_ids) {
        const auto it = objects_.find(id);
        if (id == 0 || it == objects_.end()) { close_fd(); throw std::runtime_error("KV disk object-map lost new object"); }
        const ObjectRef& ref = it->second;
        const auto frame = encode_pack_map_frame(PackedMapEntry{
            id, ref.kind, ref.location.segment, ref.location.offset, ref.bytes,
            ref.location.stored_bytes, ref.location.logical_bytes,
            ref.location.record_crc32c});
        std::size_t done = 0;
        while (done < frame.size()) {
            const ssize_t n = ::write(fd, frame.data() + done, frame.size() - done);
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) { close_fd(); throw std::runtime_error("failed to append KV disk object-map"); }
            done += static_cast<std::size_t>(n);
        }
    }
    try {
        maybe_test_fault(DiskFaultPoint::AfterMapAppend);
    } catch (...) {
        close_fd();
        throw;
    }
    if (::fdatasync(fd) != 0) { close_fd(); throw std::runtime_error("failed to sync KV disk object-map"); }
    try {
        maybe_test_fault(DiskFaultPoint::AfterMapSync);
    } catch (...) {
        close_fd();
        throw;
    }
    close_fd();
}

std::uint64_t KVDiskCache::packed_allocated_bytes() const {
    std::uint64_t total = 0;
    if (!active_generation_) { return std::numeric_limits<std::uint64_t>::max(); }
    std::error_code ec;
    const auto packs = config_.location / "packs";
    if (!std::filesystem::exists(packs, ec)) {
        return ec ? std::numeric_limits<std::uint64_t>::max() : 0;
    }
    for (const auto& generation : std::filesystem::directory_iterator(packs, ec)) {
        if (ec) { return std::numeric_limits<std::uint64_t>::max(); }
        if (!generation.is_directory(ec) || ec) {
            if (ec) { return std::numeric_limits<std::uint64_t>::max(); }
            continue;
        }
        for (DiskObjectKind kind : {DiskObjectKind::Main, DiskObjectKind::Backend,
                                    DiskObjectKind::State, DiskObjectKind::Ledger,
                                    DiskObjectKind::Identity}) {
            const auto directory = generation.path() / kind_dir(kind);
            if (!std::filesystem::exists(directory, ec)) {
                if (ec) { return std::numeric_limits<std::uint64_t>::max(); }
                continue;
            }
            for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
                if (ec) { return std::numeric_limits<std::uint64_t>::max(); }
                if (!entry.is_regular_file(ec) || ec) {
                    if (ec) { return std::numeric_limits<std::uint64_t>::max(); }
                    continue;
                }
                const std::uint64_t bytes = entry.file_size(ec);
                if (ec || std::numeric_limits<std::uint64_t>::max() - total < bytes) {
                    return std::numeric_limits<std::uint64_t>::max();
                }
                total += bytes;
            }
        }
    }
    return total;
}

std::uint64_t KVDiskCache::packed_retained_bytes() const {
    std::uint64_t total = 0;
    for (const auto& [id, ref] : objects_) {
        (void)id;
        if (ref.live_refs == 0 && ref.skip_refs == 0 && ref.hold_refs == 0) { continue; }
        if (std::numeric_limits<std::uint64_t>::max() - total < ref.bytes) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        total += ref.bytes;
    }
    return total;
}

bool KVDiskCache::packs_need_compaction() const {
    const std::uint64_t allocated = packed_allocated_bytes();
    const std::uint64_t retained = packed_retained_bytes();
    return allocated != 0 && retained < allocated && allocated - retained >= allocated / 4;
}

bool KVDiskCache::compact_packs(std::unique_lock<std::mutex>& lock,
                                std::uint64_t additional_bytes) {
    if (!packs_need_compaction()) { return true; }
    // Generation descriptor replacement is safe only between restore epochs.
    // The normal idle path defers maintenance while any page/state operation is
    // in flight; emergency spill reports no durable capacity rather than racing
    // a reader with deletion of its source generation.
    if (spill_ || restore_active_ || restore_target_ || payload_io_inflight_ != 0 ||
        window_inflight_ != 0 || restore_state_inflight_ != 0 || !restore_q_.empty() ||
        !reader_claims_.empty()) {
        return false;
    }
    const auto available_bytes = [&]() -> std::optional<std::uint64_t> {
        if (test_free_bytes_override_) { return *test_free_bytes_override_; }
        struct statvfs fs {};
        if (::statvfs(config_.location.c_str(), &fs) != 0 || fs.f_frsize == 0 ||
            fs.f_bavail > std::numeric_limits<std::uint64_t>::max() / fs.f_frsize) {
            return std::nullopt;
        }
        return static_cast<std::uint64_t>(fs.f_bavail) * fs.f_frsize;
    }();
    const std::uint64_t retained = packed_retained_bytes();
    constexpr std::uint64_t kSegmentSlack = 1ULL << 30;
    if (!available_bytes || retained > std::numeric_limits<std::uint64_t>::max() - additional_bytes ||
        retained + additional_bytes > std::numeric_limits<std::uint64_t>::max() - kSegmentSlack ||
        *available_bytes < retained + additional_bytes + kSegmentSlack) {
        return false;
    }
    const std::uint64_t old_generation = pack_generation_;
    if (old_generation == std::numeric_limits<std::uint64_t>::max()) { return false; }
    std::uint64_t new_generation = old_generation + 1;
    std::error_code generation_ec;
    while (std::filesystem::exists(config_.location / "packs" / std::to_string(new_generation),
                                   generation_ec)) {
        if (generation_ec || new_generation == std::numeric_limits<std::uint64_t>::max()) {
            return false;
        }
        ++new_generation;
    }
    if (generation_ec) { return false; }
    struct Move { std::uint64_t id; ObjectRef::Location location; };
    std::vector<Move> moves;
    moves.reserve(objects_.size());
    std::array<std::uint32_t, 5> segment{};
    std::array<std::uint64_t, 5> tail{};
    std::array<int, 5> output{};
    output.fill(-1);
    auto close_outputs = [&] {
        for (int& fd : output) { if (fd >= 0) { (void)::close(fd); fd = -1; } }
    };
    const auto new_root = config_.location / "packs" / std::to_string(new_generation);
    auto new_active = std::make_shared<PackGeneration>();
    new_active->number = new_generation;
    new_active->root = new_root;
    bool published = false;
    try {
        for (DiskObjectKind kind : {DiskObjectKind::Main, DiskObjectKind::Backend,
                                    DiskObjectKind::State, DiskObjectKind::Ledger,
                                    DiskObjectKind::Identity}) {
            std::filesystem::create_directories(new_root / kind_dir(kind));
        }
        std::vector<std::uint64_t> ids;
        ids.reserve(objects_.size());
        for (const auto& [id, ref] : objects_) {
            if (ref.live_refs != 0 || ref.skip_refs != 0 || ref.hold_refs != 0) { ids.push_back(id); }
        }
        std::sort(ids.begin(), ids.end());
        std::vector<std::uint8_t> buffer(1U << 20);
        const auto read_fully = [](int fd, void* data, std::size_t bytes, off_t offset) {
            auto* out = static_cast<std::uint8_t*>(data);
            std::size_t done = 0;
            while (done < bytes) {
                const ssize_t n = ::pread(fd, out + done, bytes - done,
                                          offset + static_cast<off_t>(done));
                if (n < 0 && errno == EINTR) { continue; }
                if (n <= 0) { return false; }
                done += static_cast<std::size_t>(n);
            }
            return true;
        };
        const auto write_fully = [](int fd, const void* data, std::size_t bytes, off_t offset) {
            const auto* in = static_cast<const std::uint8_t*>(data);
            std::size_t done = 0;
            while (done < bytes) {
                const ssize_t n = ::pwrite(fd, in + done, bytes - done,
                                           offset + static_cast<off_t>(done));
                if (n < 0 && errno == EINTR) { continue; }
                if (n <= 0) { return false; }
                done += static_cast<std::size_t>(n);
            }
            return true;
        };
        for (std::uint64_t id : ids) {
            const ObjectRef& ref = objects_.at(id);
            const std::size_t index = kind_index(ref.kind);
            if (!ref.location.generation || ref.location.stored_bytes == 0 ||
                ref.location.stored_bytes > ref.bytes || ref.location.logical_bytes == 0) {
                throw std::runtime_error("invalid KV pack source metadata during compaction id=" +
                                         std::to_string(id) + " kind=" +
                                         std::to_string(static_cast<unsigned>(ref.kind)) +
                                         " extent=" + std::to_string(ref.bytes) +
                                         " stored=" + std::to_string(ref.location.stored_bytes) +
                                         " logical=" + std::to_string(ref.location.logical_bytes));
            }
            if (tail[index] != 0 && tail[index] + ref.bytes > kPackSegmentBytes) {
                ++segment[index]; tail[index] = 0;
                if (output[index] >= 0) {
                    const int fd = output[index];
                    output[index] = -1;
                    if (::fdatasync(fd) != 0) {
                        (void)::close(fd);
                        throw std::runtime_error("failed to sync rolled compacted KV pack");
                    }
                    if (::close(fd) != 0) {
                        throw std::runtime_error("failed to close rolled compacted KV pack");
                    }
                }
            }
            const auto out_path = new_root / kind_dir(ref.kind) / ([&] {
                char name[32]{}; std::snprintf(name, sizeof(name), "%08u.pack", segment[index]);
                return std::string(name);
            })();
            if (output[index] < 0) {
                std::filesystem::create_directories(out_path.parent_path());
                output[index] = ::open(out_path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
                if (output[index] < 0) { throw std::runtime_error("failed to create compacted KV pack"); }
            }
            const auto input = acquire_pack_descriptor(ref.location.generation, ref.kind,
                                                       ref.location.segment, false);
            struct stat input_stat {};
            if (::fstat(input->buffered, &input_stat) != 0 ||
                static_cast<std::uint64_t>(input_stat.st_size) <
                    ref.location.offset + ref.bytes) {
                throw std::runtime_error("short KV pack source during compaction");
            }
            std::uint8_t header[kDiskStatePayloadOffset]{};
            const std::size_t header_bytes = ref.kind == DiskObjectKind::State
                                                 ? kDiskCodecHeaderBytes
                                                 : (ref.kind == DiskObjectKind::Main ||
                                                    ref.kind == DiskObjectKind::Backend
                                                        ? kDiskPageHeaderBytes
                                                        : 0);
            if (header_bytes != 0 &&
                !read_fully(input->buffered, header, header_bytes,
                            static_cast<off_t>(ref.location.offset))) {
                throw std::runtime_error("failed to read KV pack source header");
            }
            std::uint32_t record_crc = ~0U;
            std::uint32_t payload_crc = ~0U;
            std::uint64_t copied = 0;
            while (copied < ref.bytes) {
                const std::size_t chunk = static_cast<std::size_t>(std::min<std::uint64_t>(
                    buffer.size(), ref.bytes - copied));
                if (!read_fully(input->buffered, buffer.data(), chunk,
                                static_cast<off_t>(ref.location.offset + copied)) ||
                    !write_fully(output[index], buffer.data(), chunk,
                                 static_cast<off_t>(tail[index] + copied))) {
                    throw std::runtime_error("failed to copy KV pack extent");
                }
                const std::uint64_t stored_begin = copied;
                const std::uint64_t stored_end = std::min<std::uint64_t>(
                    copied + chunk, ref.location.stored_bytes);
                if (stored_begin < stored_end) {
                    const auto stored = std::span<const std::uint8_t>(
                        buffer.data(), static_cast<std::size_t>(stored_end - stored_begin));
                    record_crc = crc32c_update(record_crc, stored);
                    const std::uint64_t payload_begin =
                        ref.kind == DiskObjectKind::State ? kDiskStatePayloadOffset
                        : (ref.kind == DiskObjectKind::Main || ref.kind == DiskObjectKind::Backend
                               ? kDiskPageHeaderBytes
                               : 0);
                    if (stored_begin < ref.location.stored_bytes &&
                        payload_begin < ref.location.stored_bytes) {
                        const std::uint64_t begin = std::max(stored_begin, payload_begin);
                        const std::uint64_t end = stored_end;
                        if (begin < end) {
                            payload_crc = crc32c_update(
                                payload_crc,
                                std::span<const std::uint8_t>(
                                    buffer.data() + static_cast<std::size_t>(begin - copied),
                                    static_cast<std::size_t>(end - begin)));
                        }
                    }
                }
                copied += chunk;
            }
            if (~record_crc != ref.location.record_crc32c) {
                throw std::runtime_error("KV pack record CRC mismatch during compaction");
            }
            if (ref.kind == DiskObjectKind::Main || ref.kind == DiskObjectKind::Backend) {
                std::uint32_t version = 0;
                std::uint32_t expected = 0;
                std::memcpy(&version, header + 8, sizeof(version));
                std::memcpy(&expected, header + 12, sizeof(expected));
                if (std::memcmp(header, kDiskPageMagic, 8) != 0 ||
                    version != kDiskFormatVersion || ref.location.stored_bytes !=
                        kDiskPageHeaderBytes + ref.location.logical_bytes ||
                    ref.bytes != aligned_page_file_bytes(ref.location.logical_bytes) ||
                    expected != ~payload_crc) {
                    throw std::runtime_error("invalid KV page record during compaction");
                }
            } else if (ref.kind == DiskObjectKind::State) {
                const auto codec = static_cast<DiskCodec>(header[0]);
                const auto state_kind = static_cast<DiskStateKind>(header[1]);
                std::uint64_t unc = 0;
                std::uint64_t cmp = 0;
                std::uint32_t expected = 0;
                std::memcpy(&unc, header + 4, sizeof(unc));
                std::memcpy(&cmp, header + 12, sizeof(cmp));
                std::memcpy(&expected, header + 20, sizeof(expected));
                if (!valid_state_kind(static_cast<std::uint8_t>(state_kind)) ||
                    !codec_payload_ok(codec, unc, cmp, ref.location.logical_bytes) ||
                    unc != ref.location.logical_bytes ||
                    cmp != ref.location.stored_bytes - kDiskStatePayloadOffset ||
                    expected != ~payload_crc) {
                    throw std::runtime_error("invalid KV state record during compaction");
                }
            } else if (ref.location.logical_bytes != ref.location.stored_bytes) {
                throw std::runtime_error("invalid KV raw record during compaction");
            }
            moves.push_back(Move{id, ObjectRef::Location{nullptr, segment[index], tail[index], ref.bytes,
                                                          ref.location.stored_bytes,
                                                          ref.location.logical_bytes,
                                                          ref.location.record_crc32c}});
            tail[index] += ref.bytes;
        }
        maybe_test_fault(DiskFaultPoint::CompactionAfterPackCopy);
        for (int fd : output) {
            if (fd >= 0 && ::fdatasync(fd) != 0) { throw std::runtime_error("failed to sync compacted KV pack"); }
        }
        maybe_test_fault(DiskFaultPoint::CompactionAfterPackSync);
        close_outputs();
        for (DiskObjectKind kind : {DiskObjectKind::Main, DiskObjectKind::Backend,
                                    DiskObjectKind::State, DiskObjectKind::Ledger,
                                    DiskObjectKind::Identity}) {
            const std::size_t index = kind_index(kind);
            const auto path = pack_path(*new_active, kind, segment[index]);
            if (std::filesystem::exists(path)) { continue; }
            const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
            if (fd < 0) { throw std::runtime_error("failed to create compacted KV pack segment"); }
            if (::fsync(fd) != 0) {
                ::close(fd);
                throw std::runtime_error("failed to sync compacted KV pack segment");
            }
            ::close(fd);
        }
        std::vector<PackedMapEntry> base_entries;
        base_entries.reserve(moves.size());
        for (const Move& move : moves) {
            const ObjectRef& ref = objects_.at(move.id);
            base_entries.push_back({move.id, ref.kind, move.location.segment, move.location.offset,
                                    ref.bytes, move.location.stored_bytes,
                                    move.location.logical_bytes,
                                    move.location.record_crc32c});
        }
        const auto base = config_.location / "maps" / ("objects-" + std::to_string(new_generation) + ".base");
        const auto log = config_.location / "maps" / ("objects-" + std::to_string(new_generation) + ".log");
        const auto base_bytes = encode_pack_map_base(new_generation, next_object_id_, std::move(base_entries));
        write_file_bytes(base, base_bytes.data(), base_bytes.size());
        maybe_test_fault(DiskFaultPoint::CompactionAfterBaseWrite);
        fsync_path(base);
        maybe_test_fault(DiskFaultPoint::CompactionAfterBaseSync);
        write_file_bytes(log, nullptr, 0);
        maybe_test_fault(DiskFaultPoint::CompactionAfterLogWrite);
        fsync_path(log);
        maybe_test_fault(DiskFaultPoint::CompactionAfterLogSync);
        fsync_dir(new_root / "main"); fsync_dir(new_root / "backend"); fsync_dir(new_root / "state");
        fsync_dir(new_root / "ledger"); fsync_dir(new_root / "identity"); fsync_dir(new_root);
        fsync_dir(config_.location / "packs"); fsync_dir(config_.location / "maps");
        const auto tmp = config_.location / "tmp" / "PACKSET";
        const auto bytes = encode_packset(new_generation, object_id_reservation_limit_);
        write_file_bytes(tmp, bytes.data(), bytes.size());
        maybe_test_fault(DiskFaultPoint::CompactionAfterPacksetTmpWrite);
        fsync_path(tmp);
        maybe_test_fault(DiskFaultPoint::CompactionAfterPacksetTmpSync);
        std::filesystem::rename(tmp, config_.location / "PACKSET");
        published = true;
        packset_publication_pending_sync_ = true;
        // Rename success is an uncertain publication until root fsync.  The
        // visible in-process generation must follow the rename immediately;
        // the old generation remains leased and retained on a root-fsync
        // failure because restart may observe either pointer.
        pack_generation_ = new_generation;
        pack_active_segment_ = segment; pack_active_tail_ = tail;
        if (active_generation_) {
            retired_generations_.push_back(std::move(active_generation_));
        }
        active_generation_ = new_active;
        for (const Move& move : moves) {
            objects_.at(move.id).location = move.location;
            objects_.at(move.id).location.generation = active_generation_;
        }
        // Unreferenced map entries are physical garbage, not leases.  Drop
        // their old-generation pointers after the new map is active so the
        // retired generation can be reclaimed once real readers have drained.
        for (auto it = objects_.begin(); it != objects_.end();) {
            const ObjectRef& ref = it->second;
            if (ref.live_refs == 0 && ref.skip_refs == 0 && ref.hold_refs == 0) {
                it = objects_.erase(it);
            } else {
                ++it;
            }
        }
        maybe_test_fault(DiskFaultPoint::CompactionAfterPacksetRename);
        fsync_dir(config_.location);
        packset_publication_pending_sync_ = false;
        maybe_test_fault(DiskFaultPoint::CompactionAfterRootSync);
    } catch (...) {
        close_outputs();
        if (!published) {
            std::error_code cleanup_ec;
            std::filesystem::remove_all(new_root, cleanup_ec);
            (void)std::filesystem::remove(
                config_.location / "maps" / ("objects-" + std::to_string(new_generation) + ".base"),
                cleanup_ec);
            (void)std::filesystem::remove(
                config_.location / "maps" / ("objects-" + std::to_string(new_generation) + ".log"),
                cleanup_ec);
        }
        // A successful PACKSET rename followed by a failed root fsync is an
        // uncertain publication.  Keep the new generation active and retain
        // the old lease; the caller reports maintenance failure.
        return false;
    }
    reap_retired_generations();
    return true;
}

void KVDiskCache::reap_retired_generations() {
    if (packset_publication_pending_sync_) { return; }
    for (auto it = retired_generations_.begin(); it != retired_generations_.end();) {
        const auto& generation = *it;
        if (!generation || generation.use_count() != 1) {
            ++it;
            continue;
        }
        const auto packs = config_.location / "packs";
        const auto maps = config_.location / "maps";
        const auto base = maps / ("objects-" + std::to_string(generation->number) + ".base");
        const auto log = maps / ("objects-" + std::to_string(generation->number) + ".log");
        std::error_code pack_ec;
        std::error_code base_ec;
        std::error_code log_ec;
        std::filesystem::remove_all(generation->root, pack_ec);
        std::filesystem::remove(base, base_ec);
        std::filesystem::remove(log, log_ec);
        if (pack_ec || base_ec || log_ec) {
            ++it;
            continue;
        }
        try {
            // Synchronize both namespaces even when a prior attempt already
            // removed the names but failed before the directory fsyncs.
            fsync_dir(packs);
            fsync_dir(maps);
        } catch (...) {
            ++drops_;
            ++it;
            continue;
        }
        it = retired_generations_.erase(it);
    }
}

void KVDiskCache::write_manifest() const {
    OutBuf w;
    w.raw(kDiskManifestMagic, 8);
    w.u32(kDiskFormatVersion);
    w.u32(static_cast<std::uint32_t>(fifo_.size()));
    for (std::uint64_t id : fifo_) { w.u64(id); }
    manifest_io_entered_.store(true, std::memory_order_release);
    const int stall_ms = manifest_io_stall_ms_.load(std::memory_order_acquire);
    if (stall_ms > 0) { std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms)); }
    const auto path = config_.location / "MANIFEST";
    write_file_bytes(path, w.bytes.data(), w.bytes.size());
    fsync_path(path);
}

void KVDiskCache::write_manifest(std::unique_lock<std::mutex>& lock) const {
    OutBuf w;
    w.raw(kDiskManifestMagic, 8);
    w.u32(kDiskFormatVersion);
    w.u32(static_cast<std::uint32_t>(fifo_.size()));
    for (std::uint64_t id : fifo_) { w.u64(id); }
    lock.unlock();
    try {
        manifest_io_entered_.store(true, std::memory_order_release);
        const int stall_ms = manifest_io_stall_ms_.load(std::memory_order_acquire);
        if (stall_ms > 0) { std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms)); }
        const auto path = config_.location / "MANIFEST";
        write_file_bytes(path, w.bytes.data(), w.bytes.size());
        fsync_path(path);
    } catch (...) {
        if (!lock.owns_lock()) { lock.lock(); }
        throw;
    }
    if (!lock.owns_lock()) { lock.lock(); }
}

void KVDiskCache::rebuild_manifest() {
    fifo_.clear();
    adopt_object_ids();
    adopt_tombstone_ids();
    std::error_code ec;
    const std::uint32_t limit = index_entry_limit();
    BoundedIdSet keep(limit);
    for (const auto& entry : std::filesystem::directory_iterator(config_.location / "entries", ec)) {
        if (!entry.is_directory()) { continue; }
        std::uint64_t id = 0;
        try {
            id = static_cast<std::uint64_t>(std::stoull(entry.path().filename().string()));
        } catch (...) { continue; }
        if (id == 0) { continue; }
        bump_next_id(next_entry_id_, id);
        if (entry_is_tombstoned(id)) {
            queue_unlink(entry_dir(id));
            continue;
        }
        keep.consider(id);
    }
    for (std::uint64_t id : keep.take_sorted()) { (void)load_entry(id); }
}

bool KVDiskCache::load_entry(std::uint64_t entry_id) {
    try {
    if (entries_.count(entry_id) != 0) { return true; }
    const auto meta_path = entry_dir(entry_id) / "meta.bin";
    std::error_code ec;
    if (!std::filesystem::exists(meta_path, ec)) { return false; }
    const auto decoded = try_decode_meta(read_file_bytes(meta_path));
    if (!decoded) {
        ++drops_;
        return false;
    }
    const DiskMeta& meta = *decoded;
    if (meta.entry_id != 0 && meta.entry_id != entry_id) {
        ++drops_;
        return false;
    }
    if (meta_cannot_restore(meta)) {
        ++drops_;
        return false;
    }
    const bool too_long =
        config_.max_context != 0 && meta.execution_frontier > config_.max_context;
    IndexEntry record;
    record.meta = meta;
    if (meta.ledger_id != 0) {
        if (sizeof(TokenId) != 0 &&
            meta.ledger_frontier > kDiskHostFileMaxBytes / sizeof(TokenId)) {
            ++drops_;
            return false;
        }
        const std::uint64_t want =
            static_cast<std::uint64_t>(meta.ledger_frontier) * sizeof(TokenId);
        const auto object = objects_.find(meta.ledger_id);
        if (object == objects_.end() || object->second.kind != DiskObjectKind::Ledger ||
            object->second.location.stored_bytes != want) {
            ++drops_;
            return false;
        }
        if (!too_long) {
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(want));
            if (!read_object(DiskObjectKind::Ledger, meta.ledger_id, bytes.data(), want)) {
                ++drops_;
                return false;
            }
            record.ledger.resize(meta.ledger_frontier);
            std::memcpy(record.ledger.data(), bytes.data(), bytes.size());
        }
    }
    if (meta.identity_id != 0) {
        const auto object = objects_.find(meta.identity_id);
        const std::uint64_t sz = object == objects_.end() ? 0 : object->second.location.stored_bytes;
        if (sz == 0 || sz > kDiskHostFileMaxBytes) {
            ++drops_;
            return false;
        }
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(sz));
        if (object->second.kind != DiskObjectKind::Identity ||
            !read_object(DiskObjectKind::Identity, meta.identity_id, bytes.data(), sz)) {
            ++drops_;
            return false;
        }
        ResidentPrefixIdentity identity;
        try {
            identity.unpack(bytes.data(), bytes.size());
        } catch (...) {
            ++drops_;
            return false;
        }
        if (identity.size() != static_cast<std::size_t>(meta.ledger_frontier)) {
            ++drops_;
            return false;
        }
        if (!too_long) { record.identity = std::move(identity); }
    }
    std::mutex startup_validation_mu;
    auto header_ok = [&](DiskObjectKind kind, std::uint64_t id, bool page,
                         std::optional<DiskStateKind> expected = std::nullopt) {
        if (id == 0) { return true; }
        const auto object = objects_.find(id);
        const std::uint64_t sz = object == objects_.end() ? 0 : object->second.bytes;
        if (object == objects_.end() || object->second.kind != kind) { return false; }
        {
            std::lock_guard validation_lock(startup_validation_mu);
            if (const auto cached = startup_validated_objects_.find(id);
                cached != startup_validated_objects_.end()) {
                return cached->second.kind == kind && cached->second.page == page &&
                       (!expected || cached->second.state_kind == expected);
            }
        }
        std::error_code size_ec;
        const std::uint64_t pack_bytes = std::filesystem::file_size(
            pack_path(*object->second.location.generation, kind,
                      object->second.location.segment), size_ec);
        if (size_ec || object->second.location.offset > pack_bytes ||
            sz > pack_bytes - object->second.location.offset) {
            return false;
        }
        if (page) {
            std::size_t payload = config_.logical_page_bytes;
            if (kind == DiskObjectKind::Main && config_.text_pool != nullptr) {
                payload = paged_kv_logical_page_bytes(*config_.text_pool);
            } else if (kind == DiskObjectKind::Backend && config_.backend_pool != nullptr) {
                payload = paged_kv_logical_page_bytes(*config_.backend_pool);
            }
            if (!payload_fits(sz, kDiskPageHeaderBytes, payload)) { return false; }
            std::vector<std::uint8_t> stored(kDiskPageHeaderBytes + payload);
            if (!read_object(kind, id, stored.data(), stored.size()) ||
                std::memcmp(stored.data(), kDiskPageMagic, 8) != 0) {
                return false;
            }
            std::uint32_t version = 0;
            std::uint32_t expected_crc = 0;
            std::memcpy(&version, stored.data() + 8, 4);
            std::memcpy(&expected_crc, stored.data() + 12, 4);
            const bool valid = version == kDiskFormatVersion &&
                               expected_crc == crc32c(std::span<const std::uint8_t>(
                                                   stored.data() + kDiskPageHeaderBytes, payload));
            if (valid) {
                std::lock_guard validation_lock(startup_validation_mu);
                startup_validated_objects_.emplace(
                    id, StartupObjectValidation{kind, true, std::nullopt});
            }
            return valid;
        }
        if (sz < kDiskCodecHeaderBytes) { return false; }
        std::vector<std::uint8_t> hdr(kDiskCodecHeaderBytes);
        if (!read_object(kind, id, hdr.data(), hdr.size())) { return false; }
        if (hdr[0] > 1 || !valid_state_kind(hdr[1])) { return false; }
        const auto actual_kind = static_cast<DiskStateKind>(hdr[1]);
        if (expected && actual_kind != *expected) { return false; }
        std::uint64_t unc = 0;
        std::uint64_t cmp = 0;
        std::memcpy(&unc, hdr.data() + 4, 8);
        std::memcpy(&cmp, hdr.data() + 12, 8);
        const std::uint64_t want =
            expected_state_bytes(config_.fingerprint, actual_kind, config_.hidden_bytes);
        if (want != 0 && unc != want) { return false; }
        const std::uint64_t max_unc = want != 0 ? want : kDiskHostFileMaxBytes;
        if (!codec_payload_ok(static_cast<DiskCodec>(hdr[0]), unc, cmp, max_unc)) {
            return false;
        }
        if (!payload_fits(sz, kDiskStatePayloadOffset, cmp)) { return false; }
        std::vector<std::uint8_t> stored(static_cast<std::size_t>(kDiskStatePayloadOffset + cmp));
        std::uint32_t expected_crc = 0;
        std::memcpy(&expected_crc, hdr.data() + 20, 4);
        const bool valid = read_object(kind, id, stored.data(), stored.size()) &&
                           expected_crc == crc32c(std::span<const std::uint8_t>(
                               stored.data() + kDiskStatePayloadOffset,
                               static_cast<std::size_t>(cmp)));
        if (valid) {
            std::lock_guard validation_lock(startup_validation_mu);
            startup_validated_objects_.emplace(
                id, StartupObjectValidation{kind, false, actual_kind});
        }
        return valid;
    };
    const bool need_gdn = config_.fingerprint.gdn_conv_bytes != 0 ||
                          config_.fingerprint.gdn_recurrent_bytes != 0;
    const bool need_cyclic = config_.fingerprint.cyclic_lane_bytes != 0;
    auto slot_missing = [&](const DiskCheckpointSlot& slot) {
        if (slot.frontier == 0) { return false; }
        if (need_gdn && slot.gdn_id == 0) { return true; }
        if (need_cyclic && slot.dflash_id == 0) { return true; }
        return false;
    };
    if ((need_gdn && meta.current_gdn_id == 0) || (need_cyclic && meta.current_cyclic_id == 0) ||
        (need_gdn && meta.rewrite_valid && meta.rewrite_gdn_id == 0) ||
        (need_cyclic && meta.rewrite_valid && meta.rewrite_cyclic_id == 0) ||
        slot_missing(meta.rollback) || slot_missing(meta.ladders[0]) ||
        slot_missing(meta.ladders[1])) {
        ++drops_;
        return false;
    }
    if (need_cyclic) {
        auto need_cyclic_with_gdn = [](const DiskCheckpointSlot& slot) {
            return slot.frontier != 0 && slot.gdn_id != 0 && slot.dflash_id == 0;
        };
        if ((meta.current_gdn_id != 0 && meta.current_cyclic_id == 0) ||
            (meta.rewrite_gdn_id != 0 && meta.rewrite_cyclic_id == 0) ||
            need_cyclic_with_gdn(meta.rollback) || need_cyclic_with_gdn(meta.ladders[0]) ||
            need_cyclic_with_gdn(meta.ladders[1])) {
            ++drops_;
            return false;
        }
    }
    bool objects_ok =
        header_ok(DiskObjectKind::State, meta.current_gdn_id, false, DiskStateKind::CurrentGdn) &&
        header_ok(DiskObjectKind::State, meta.current_hidden_id, false, DiskStateKind::TailHidden) &&
        header_ok(DiskObjectKind::State, meta.current_cyclic_id, false, DiskStateKind::DflashLocal) &&
        header_ok(DiskObjectKind::State, meta.rewrite_gdn_id, false, DiskStateKind::RewriteGdn) &&
        header_ok(DiskObjectKind::State, meta.rewrite_hidden_id, false,
                  DiskStateKind::RewriteHidden) &&
        header_ok(DiskObjectKind::State, meta.rewrite_cyclic_id, false, DiskStateKind::DflashRewrite) &&
        header_ok(DiskObjectKind::State, meta.rollback.gdn_id, false, DiskStateKind::RollbackGdn) &&
        header_ok(DiskObjectKind::State, meta.rollback.hidden_id, false,
                  DiskStateKind::RollbackHidden) &&
        header_ok(DiskObjectKind::State, meta.rollback.dflash_id, false,
                  DiskStateKind::DflashRollback);
    for (const auto& slot : meta.ladders) {
        objects_ok =
            objects_ok &&
            header_ok(DiskObjectKind::State, slot.gdn_id, false, DiskStateKind::LadderGdn) &&
            header_ok(DiskObjectKind::State, slot.hidden_id, false, DiskStateKind::LadderHidden) &&
            header_ok(DiskObjectKind::State, slot.dflash_id, false, DiskStateKind::DflashLadder);
    }
    auto pages_ok = [&](DiskObjectKind kind, const std::vector<std::uint64_t>& ids) {
        if (ids.empty()) { return true; }
        std::atomic<std::size_t> next{0};
        std::atomic<bool> valid{true};
        auto run = [&] {
            for (;;) {
                const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= ids.size()) { break; }
                const std::uint64_t id = ids[index];
                if (id == 0 || !header_ok(kind, id, true)) {
                    valid.store(false, std::memory_order_relaxed);
                }
            }
        };
        const std::uint32_t workers = std::min<std::uint32_t>(
            restore_io_threads_, static_cast<std::uint32_t>(ids.size()));
        std::vector<std::thread> pool;
        pool.reserve(workers > 0 ? workers - 1 : 0);
        try {
            for (std::uint32_t i = 1; i < workers; ++i) { pool.emplace_back(run); }
            run();
        } catch (...) {
            valid.store(false, std::memory_order_relaxed);
        }
        for (auto& thread : pool) {
            if (thread.joinable()) { thread.join(); }
        }
        return valid.load(std::memory_order_relaxed);
    };
    objects_ok = objects_ok && pages_ok(DiskObjectKind::Main, meta.main_page_ids);
    objects_ok = objects_ok && pages_ok(DiskObjectKind::Backend, meta.backend_page_ids);
    const std::uint32_t kv_tokens = std::max(meta.execution_frontier, meta.text_kv_valid);
    const std::uint32_t need_main = ninfer::pages_for_tokens(kv_tokens);
    const std::uint32_t need_backend = ninfer::pages_for_tokens(backend_tokens_required(meta));
    if (meta.main_page_ids.size() < need_main || meta.backend_page_ids.size() < need_backend) {
        objects_ok = false;
    }
    if (!objects_ok) {
        ++drops_;
        return false;
    }
    std::vector<std::pair<DiskObjectKind, std::uint64_t>> owned;
    append_meta_objects(owned, meta);
    if (owned_id_kind_conflict(owned)) {
        ++drops_;
        return false;
    }
    for (const auto& [kind, id] : owned) {
        if (object_kind_conflict(kind, id)) {
            ++drops_;
            return false;
        }
    }
    if (too_long) {
        SkippedTree skip;
        skip.entry_id = entry_id;
        skip.mtime    = std::filesystem::last_write_time(meta_path, ec);
        skip.objects  = owned;
        for (const auto& [kind, id] : owned) {
            objects_[id].kind = kind;
            add_skip_ref(id);
            bump_next_id(next_object_id_, id);
        }
        skipped_.push_back(std::move(skip));
        bump_next_id(next_entry_id_, entry_id);
        return false;
    }
    auto bump = [&](DiskObjectKind kind, std::uint64_t id) {
        if (id == 0) { return; }
        objects_[id].kind = kind;
        add_ref(id);
        bump_next_id(next_object_id_, id);
    };
    for (const auto& [kind, id] : owned) { bump(kind, id); }
    if (record.committed_generation == 0) { record.committed_generation = 1; }
    entries_.emplace(entry_id, std::move(record));
    fifo_.push_back(entry_id);
    bump_next_id(next_entry_id_, entry_id);
    return true;
    } catch (...) {
        ++drops_;
        return false;
    }
}

void KVDiskCache::load_index() {
    startup_loading_index_ = true;
    struct StartupLoadGuard {
        bool& loading;
        std::unordered_map<std::uint64_t, StartupObjectValidation>& validated;
        ~StartupLoadGuard() {
            validated.clear();
            validated.rehash(0);
            loading = false;
        }
    } startup_guard{startup_loading_index_, startup_validated_objects_};
    adopt_object_ids();
    adopt_tombstone_ids();
    const auto path = config_.location / "MANIFEST";
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        rebuild_manifest();
    } else {
        try {
            const auto bytes = read_file_bytes(path);
            if (bytes.size() < 16 || std::memcmp(bytes.data(), kDiskManifestMagic, 8) != 0) {
                throw std::runtime_error("torn");
            }
            InBuf r{bytes.data() + 8, bytes.data() + bytes.size()};
            if (r.u32() != kDiskFormatVersion) { throw std::runtime_error("unknown"); }
            const auto n = r.u32();
            const std::uint32_t limit = index_entry_limit();
            if (n > limit) { throw std::runtime_error("torn"); }
            const std::uint64_t need = static_cast<std::uint64_t>(n) * 8ULL;
            if (need / 8ULL != n || r.remain() < need) { throw std::runtime_error("torn"); }
            std::vector<std::uint64_t> ids(n);
            for (std::uint32_t i = 0; i < n; ++i) { ids[i] = r.u64(); }
            std::unordered_set<std::uint64_t> listed(ids.begin(), ids.end());
            if (listed.size() != ids.size()) { throw std::runtime_error("torn"); }
            for (std::uint64_t id : ids) {
                bump_next_id(next_entry_id_, id);
                if (entry_is_tombstoned(id)) {
                    queue_unlink(entry_dir(id));
                    continue;
                }
                (void)load_entry(id);
            }
            bool extras = false;
            const std::uint32_t resident =
                static_cast<std::uint32_t>(entries_.size() + skipped_.size());
            const std::uint32_t remain =
                resident >= limit ? 0 : static_cast<std::uint32_t>(limit - resident);
            BoundedIdSet extra_keep(remain);
            for (const auto& entry :
                 std::filesystem::directory_iterator(config_.location / "entries", ec)) {
                if (!entry.is_directory()) { continue; }
                std::uint64_t id = 0;
                try {
                    id = static_cast<std::uint64_t>(std::stoull(entry.path().filename().string()));
                } catch (...) { continue; }
                if (id == 0) { continue; }
                bump_next_id(next_entry_id_, id);
                if (listed.count(id) != 0) { continue; }
                if (entry_is_tombstoned(id)) {
                    queue_unlink(entry_dir(id));
                    continue;
                }
                extra_keep.consider(id);
            }
            for (std::uint64_t id : extra_keep.take_sorted()) {
                if (entries_.size() + skipped_.size() >= limit) { continue; }
                extras = true;
                (void)load_entry(id);
            }
            if (extras) {
                try {
                    write_manifest();
                } catch (...) {}
            }
            queue_tombstone_object_unlinks();
            flush_queued_unlinks();
            try {
                fsync_store_dirs();
            } catch (...) {}
            reclaim_durable_tombstones();
        } catch (...) { rebuild_manifest(); }
    }
    bool trimmed = false;
    while (unique_bytes_ > config_.capacity_bytes) {
        bool evicted = false;
        for (std::uint64_t id : fifo_) {
            auto it = entries_.find(id);
            if (it != entries_.end() && !it->second.pinned && it->second.io_pins == 0) {
                fifo_evict_one();
                evicted = true;
                break;
            }
        }
        if (!evicted) { break; }
        trimmed = true;
    }
    queue_tombstone_object_unlinks();
    flush_queued_unlinks();
    if (trimmed) {
        try {
            write_manifest();
        } catch (...) { ++drops_; }
        try {
            fsync_store_dirs();
        } catch (...) { ++drops_; }
        reclaim_durable_tombstones();
    }
    reclaim_orphan_objects();
    flush_queued_unlinks();
}

KVDiskCache::IndexEntry& KVDiskCache::require(std::uint64_t entry_id) {
    const auto it = entries_.find(entry_id);
    if (it == entries_.end()) { throw std::logic_error("KV disk entry is missing"); }
    return it->second;
}

const KVDiskCache::IndexEntry& KVDiskCache::require(std::uint64_t entry_id) const {
    const auto it = entries_.find(entry_id);
    if (it == entries_.end()) { throw std::logic_error("KV disk entry is missing"); }
    return it->second;
}

void KVDiskCache::pin_disk(std::uint64_t entry_id) {
    IndexEntry& record = require(entry_id);
    ++record.io_pins;
}

void KVDiskCache::unpin_disk(std::uint64_t entry_id) {
    auto it = entries_.find(entry_id);
    if (it == entries_.end()) { return; }
    if (it->second.io_pins == 0) { return; }
    --it->second.io_pins;
}

void KVDiskCache::invalidate_ram_notes_for_disk_entry(std::uint64_t disk_id) {
    if (disk_id == 0) { return; }
    for (auto& [ram_id, note] : ram_notes_) {
        if (note.ticket != disk_id) { continue; }
        note.durable = false;
        note.ticket  = 0;
        if (config_.ram != nullptr) {
            try {
                config_.ram->set_disk_entry_id(ram_id, 0);
            } catch (...) {}
        }
    }
}

void KVDiskCache::add_ref(std::uint64_t id) {
    if (id == 0) { return; }
    ObjectRef& ref = objects_[id];
    if (ref.live_refs == 0) {
        if (ref.bytes == 0) { throw std::runtime_error("KV disk object location is missing"); }
        unique_bytes_ += ref.bytes;
    }
    ++ref.live_refs;
}

void KVDiskCache::add_skip_ref(std::uint64_t id) {
    if (id == 0) { return; }
    ObjectRef& ref = objects_[id];
    if (ref.bytes == 0) { throw std::runtime_error("KV disk object location is missing"); }
    ++ref.skip_refs;
}

void KVDiskCache::add_hold(std::uint64_t id) {
    if (id == 0) { return; }
    ObjectRef& ref = objects_[id];
    if (ref.bytes == 0) { throw std::runtime_error("KV disk object location is missing"); }
    ++ref.hold_refs;
}

void KVDiskCache::release_live_ref(std::uint64_t id) {
    if (id == 0) { return; }
    const auto it = objects_.find(id);
    if (it == objects_.end() || it->second.live_refs == 0) { return; }
    --it->second.live_refs;
    if (it->second.live_refs == 0) { unique_bytes_ -= it->second.bytes; }
}

void KVDiskCache::release_hold(std::uint64_t id) {
    if (id == 0) { return; }
    const auto it = objects_.find(id);
    if (it == objects_.end() || it->second.hold_refs == 0) { return; }
    --it->second.hold_refs;
    unlink_unreferenced(id);
}

void KVDiskCache::drop_ref(std::uint64_t id) {
    release_live_ref(id);
    unlink_unreferenced(id);
}

void KVDiskCache::drop_skip_ref(std::uint64_t id) {
    if (id == 0) { return; }
    const auto it = objects_.find(id);
    if (it == objects_.end() || it->second.skip_refs == 0) { return; }
    --it->second.skip_refs;
    unlink_unreferenced(id);
}

void KVDiskCache::queue_unlink(const std::filesystem::path& path) {
    pending_unlinks_.push_back(path);
}

bool KVDiskCache::unlink_path(const std::filesystem::path& path) {
    auto gone = [&](const std::filesystem::path& p) {
        std::error_code exists_ec;
        return !std::filesystem::exists(p, exists_ec) && !exists_ec;
    };
    auto remove_ok = [&](const std::filesystem::path& p) {
        std::error_code rec;
        std::filesystem::remove_all(p, rec);
        return !rec && gone(p);
    };
    if (fail_object_unlink_ && path.parent_path().parent_path().filename() == "objects") {
        return false;
    }
    std::error_code ec;
    if (path.parent_path().filename() == "entries") {
        std::uint64_t id = 0;
        try {
            id = static_cast<std::uint64_t>(std::stoull(path.filename().string()));
        } catch (...) { id = 0; }
        if (id == 0) { return remove_ok(path); }
        if (!entry_is_tombstoned(id)) {
            std::vector<std::pair<DiskObjectKind, std::uint64_t>> objects;
            const auto meta_path = path / "meta.bin";
            if (std::filesystem::exists(meta_path, ec)) {
                try {
                    const auto decoded = try_decode_meta(read_file_bytes(meta_path));
                    if (decoded) { append_meta_objects(objects, *decoded); }
                } catch (...) {}
            }
            try {
                write_entry_tombstone(id, objects);
            } catch (...) { return false; }
        }
        if (std::filesystem::is_directory(path, ec)) {
            const auto trash = config_.location / "tmp" /
                                ("dead-" + path.filename().string() + "-" +
                                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(config_.location / "tmp", ec);
            std::filesystem::rename(path, trash, ec);
            if (!ec) { return remove_ok(trash) && gone(path); }
            std::filesystem::remove(path / "meta.bin", ec);
        }
        return remove_ok(path);
    }
    return remove_ok(path);
}

std::filesystem::path KVDiskCache::tombstone_path(std::uint64_t id) const {
    return config_.location / "tombstones" / std::to_string(id);
}

void KVDiskCache::write_entry_tombstone(
    std::uint64_t id, const std::vector<std::pair<DiskObjectKind, std::uint64_t>>& objects) {
    if (fail_tombstone_) {
        fail_tombstone_ = false;
        throw std::runtime_error("injected tombstone write failure");
    }
    const auto dir = config_.location / "tombstones";
    std::filesystem::create_directories(dir);
    std::filesystem::create_directories(config_.location / "tmp");
    const auto path  = tombstone_path(id);
    const auto tmp   = config_.location / "tmp" / ("tombstone-" + std::to_string(id));
    const auto bytes = encode_tombstone(objects);
    write_file_bytes(tmp, bytes.data(), bytes.size());
    maybe_test_fault(DiskFaultPoint::AfterTombstoneTmpWrite);
    fsync_path(tmp);
    maybe_test_fault(DiskFaultPoint::AfterTombstoneTmpSync);
    std::filesystem::rename(tmp, path);
    maybe_test_fault(DiskFaultPoint::AfterTombstoneRename);
    fsync_dir(dir);
    maybe_test_fault(DiskFaultPoint::AfterTombstoneDirSync);
}

void KVDiskCache::clear_entry_tombstone(std::uint64_t id) const {
    std::error_code ec;
    std::filesystem::remove(tombstone_path(id), ec);
}

void KVDiskCache::bump_next_id(std::uint64_t& next, std::uint64_t seen) const noexcept {
    if (seen == 0) { return; }
    if (seen == std::numeric_limits<std::uint64_t>::max()) {
        next = seen;
        return;
    }
    next = std::max(next, seen + 1);
}

void KVDiskCache::adopt_object_ids() {
    // v4 obtains the durable global high-water mark from PACKSET and the
    // object-location map before entry metadata is scanned.
}

void KVDiskCache::reclaim_durable_tombstones() {
    if (!pending_unlinks_.empty()) { return; }
    std::error_code ec;
    std::vector<std::uint64_t> drop;
    for (const auto& entry :
         std::filesystem::directory_iterator(config_.location / "tombstones", ec)) {
        if (!entry.is_regular_file()) { continue; }
        std::uint64_t id = 0;
        try {
            id = static_cast<std::uint64_t>(std::stoull(entry.path().filename().string()));
        } catch (...) { continue; }
        if (id == 0 || entries_.count(id) != 0) { continue; }
        bool skipped = false;
        for (const SkippedTree& tree : skipped_) {
            if (tree.entry_id == id) {
                skipped = true;
                break;
            }
        }
        if (skipped) { continue; }
        std::error_code exists_ec;
        if (std::filesystem::exists(entry_dir(id), exists_ec) || exists_ec) { continue; }
        std::optional<std::vector<std::pair<DiskObjectKind, std::uint64_t>>> objects;
        try {
            objects = try_decode_tombstone(read_file_bytes(entry.path()));
        } catch (...) { continue; }
        if (!objects || tombstone_objects_remain(*objects)) { continue; }
        drop.push_back(id);
    }
    for (std::uint64_t id : drop) { clear_entry_tombstone(id); }
    if (!drop.empty()) {
        try {
            fsync_dir(config_.location / "tombstones");
        } catch (...) {}
    }
}

bool KVDiskCache::object_kind_conflict(DiskObjectKind kind, std::uint64_t id) const noexcept {
    if (id == 0) { return false; }
    const auto it = objects_.find(id);
    if (it == objects_.end()) { return false; }
    return it->second.kind != kind && (it->second.live_refs != 0 || it->second.skip_refs != 0);
}

bool KVDiskCache::meta_cannot_restore(const DiskMeta& meta) const noexcept {
    if (meta.execution_frontier == std::numeric_limits<std::uint32_t>::max()) { return true; }
    if (meta.ledger_frontier != meta.execution_frontier + 1) { return true; }
    if (meta.execution_frontier != 0 && (meta.ledger_id == 0 || meta.identity_id == 0)) {
        return true;
    }
    if (meta.execution_frontier != 0 && meta.text_kv_valid < meta.execution_frontier) {
        return true;
    }
    if (meta.rewrite_valid && meta.rewrite_frontier > meta.text_kv_valid) { return true; }
    auto slot_uncovered = [&](const DiskCheckpointSlot& slot) {
        return slot.frontier != 0 && meta.text_kv_valid < slot.frontier;
    };
    return slot_uncovered(meta.rollback) || slot_uncovered(meta.ladders[0]) ||
           slot_uncovered(meta.ladders[1]);
}

std::uint32_t KVDiskCache::backend_tokens_required(const DiskMeta& meta) const noexcept {
    if (config_.backend_pool == nullptr) { return 0; }
    std::uint32_t tokens = 0;
    auto consider_mtp = [&](std::uint32_t frontier) {
        if (frontier != 0) { tokens = std::max(tokens, frontier - 1U); }
    };
    auto consider_dflash = [&](std::uint32_t frontier) {
        if (frontier != 0) { tokens = std::max(tokens, frontier); }
    };
    switch (config_.fingerprint.speculative) {
    case SpeculativeBackend::Mtp:
        tokens = meta.mtp_kv_valid;
        consider_mtp(meta.execution_frontier);
        if (meta.rewrite_valid) { consider_mtp(meta.rewrite_frontier); }
        consider_mtp(meta.rollback.frontier);
        consider_mtp(meta.ladders[0].frontier);
        consider_mtp(meta.ladders[1].frontier);
        break;
    case SpeculativeBackend::DFlash:
        tokens = meta.dflash_context_frontier != 0 ? meta.dflash_context_frontier
                                                     : meta.execution_frontier;
        consider_dflash(meta.execution_frontier);
        if (meta.rewrite_valid) { consider_dflash(meta.rewrite_frontier); }
        consider_dflash(meta.rollback.frontier);
        consider_dflash(meta.ladders[0].frontier);
        consider_dflash(meta.ladders[1].frontier);
        break;
    default:
        break;
    }
    return tokens;
}

void KVDiskCache::queue_tombstone_object_unlinks() {
    std::error_code ec;
    for (const auto& entry :
         std::filesystem::directory_iterator(config_.location / "tombstones", ec)) {
        if (!entry.is_regular_file()) { continue; }
        std::optional<std::vector<std::pair<DiskObjectKind, std::uint64_t>>> objects;
        try {
            objects = try_decode_tombstone(read_file_bytes(entry.path()));
        } catch (...) { continue; }
        if (!objects) { continue; }
        for (const auto& [kind, id] : *objects) {
            const auto it = objects_.find(id);
            if (it != objects_.end() && (it->second.live_refs != 0 || it->second.skip_refs != 0 ||
                                         it->second.hold_refs != 0)) {
                continue;
            }
            // Packed extents become reclaimable garbage after the durable
            // tombstone; they are reclaimed only by pack-set compaction.
        }
    }
}

bool KVDiskCache::tombstone_objects_remain(
    const std::vector<std::pair<DiskObjectKind, std::uint64_t>>& objects) const {
    std::error_code ec;
    for (const auto& [kind, id] : objects) {
        const auto it = objects_.find(id);
        if (it != objects_.end() && (it->second.live_refs != 0 || it->second.skip_refs != 0 ||
                                      it->second.hold_refs != 0)) {
            return true;
        }
        (void)kind;
    }
    return false;
}

void KVDiskCache::adopt_tombstone_ids() {
    std::error_code ec;
    for (const auto& entry :
         std::filesystem::directory_iterator(config_.location / "tombstones", ec)) {
        if (!entry.is_regular_file()) { continue; }
        try {
            const auto id =
                static_cast<std::uint64_t>(std::stoull(entry.path().filename().string()));
            if (id != 0) { bump_next_id(next_entry_id_, id); }
        } catch (...) { continue; }
    }
}

bool KVDiskCache::entry_is_tombstoned(std::uint64_t id) const {
    try {
        return try_decode_tombstone(read_file_bytes(tombstone_path(id))).has_value();
    } catch (...) { return false; }
}

std::uint32_t KVDiskCache::index_entry_limit() const noexcept {
    return config_.max_index_entries != 0 ? config_.max_index_entries : (1u << 18);
}

bool KVDiskCache::flush_queued_unlinks() {
    std::vector<std::filesystem::path> paths;
    paths.swap(pending_unlinks_);
    bool ok = true;
    for (const auto& path : paths) {
        try {
            if (!unlink_path(path)) {
                ok = false;
                pending_unlinks_.push_back(path);
            }
        } catch (...) {
            ok = false;
            pending_unlinks_.push_back(path);
        }
    }
    return ok;
}

bool KVDiskCache::flush_queued_unlinks(std::unique_lock<std::mutex>& lock) {
    std::vector<std::filesystem::path> paths;
    paths.swap(pending_unlinks_);
    if (paths.empty()) { return true; }
    lock.unlock();
    std::vector<std::filesystem::path> failed;
    bool ok = true;
    for (const auto& path : paths) {
        try {
            if (!unlink_path(path)) {
                ok = false;
                failed.push_back(path);
            }
        } catch (...) {
            ok = false;
            failed.push_back(path);
        }
    }
    lock.lock();
    pending_unlinks_.insert(pending_unlinks_.end(), failed.begin(), failed.end());
    return ok;
}

void KVDiskCache::unlink_unreferenced(std::uint64_t id) {
    const auto it = objects_.find(id);
    if (it == objects_.end()) { return; }
    if (it->second.live_refs != 0 || it->second.skip_refs != 0 || it->second.hold_refs != 0) {
        return;
    }
    objects_.erase(it);
}

void KVDiskCache::fifo_evict_one() {
    for (auto it = fifo_.begin(); it != fifo_.end(); ++it) {
        const auto rec = entries_.find(*it);
        if (rec == entries_.end()) { continue; }
        if (rec->second.pinned || rec->second.io_pins != 0) { continue; }
        const std::uint64_t key = *it;
        std::vector<std::pair<DiskObjectKind, std::uint64_t>> objects;
        append_meta_objects(objects, rec->second.meta);
        const std::vector<std::uint64_t> held = rec->second.uncertainty_ids;
        for (std::uint64_t id : held) {
            const auto obj = objects_.find(id);
            objects.push_back({obj == objects_.end() ? DiskObjectKind::State : obj->second.kind, id});
        }
        try {
            write_entry_tombstone(key, objects);
        } catch (...) { continue; }
        const DiskMeta meta      = rec->second.meta;
        entries_.erase(rec);
        fifo_.erase(it);
        auto drop = [&](std::uint64_t id) { drop_ref(id); };
        drop(meta.ledger_id);
        drop(meta.identity_id);
        drop(meta.current_gdn_id);
        drop(meta.current_hidden_id);
        drop(meta.current_cyclic_id);
        drop(meta.rewrite_gdn_id);
        drop(meta.rewrite_hidden_id);
        drop(meta.rewrite_cyclic_id);
        drop(meta.rollback.gdn_id);
        drop(meta.rollback.hidden_id);
        drop(meta.rollback.dflash_id);
        for (const auto& slot : meta.ladders) {
            drop(slot.gdn_id);
            drop(slot.hidden_id);
            drop(slot.dflash_id);
        }
        for (std::uint64_t id : meta.main_page_ids) { drop(id); }
        for (std::uint64_t id : meta.backend_page_ids) { drop(id); }
        for (std::uint64_t id : held) { release_hold(id); }
        queue_unlink(entry_dir(key));
        invalidate_ram_notes_for_disk_entry(key);
        ++evictions_;
        bump_version();
        bump_durable_generation();
        return;
    }
}

bool KVDiskCache::gc_skipped_one() {
    if (skipped_.empty()) { return false; }
    std::sort(skipped_.begin(), skipped_.end(),
              [](const SkippedTree& a, const SkippedTree& b) { return a.mtime < b.mtime; });
    try {
        write_entry_tombstone(skipped_.front().entry_id, skipped_.front().objects);
    } catch (...) { return false; }
    SkippedTree tree = std::move(skipped_.front());
    skipped_.erase(skipped_.begin());
    for (const auto& [kind, id] : tree.objects) {
        (void)kind;
        drop_skip_ref(id);
    }
    queue_unlink(entry_dir(tree.entry_id));
    bump_durable_generation();
    return true;
}

bool KVDiskCache::make_capacity(std::uint64_t needed, std::unique_lock<std::mutex>& lock) {
    bool trimmed = false;
    while (unique_bytes_ + needed > config_.capacity_bytes) {
        const std::uint64_t before = unique_bytes_;
        fifo_evict_one();
        if (unique_bytes_ == before) {
            if (!gc_skipped_one()) {
                const bool unlinked = flush_queued_unlinks(lock);
                if (trimmed && unlinked) { persist_eviction(lock); }
                return false;
            }
        }
        trimmed = true;
        (void)flush_queued_unlinks(lock);
    }
    const bool unlinked = flush_queued_unlinks(lock);
    if (trimmed && unlinked) { persist_eviction(lock); }
    return unique_bytes_ + needed <= config_.capacity_bytes;
}

std::uint64_t KVDiskCache::allocate_object(DiskObjectKind kind, std::uint64_t bytes) {
    if (next_object_id_ == 0 || next_object_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("KV disk object id space is exhausted");
    }
    if (next_object_id_ >= object_id_reservation_limit_) {
        constexpr std::uint64_t kReservation = 1ULL << 16;
        const std::uint64_t base = std::max(object_id_reservation_limit_, next_object_id_);
        if (base > std::numeric_limits<std::uint64_t>::max() - kReservation) {
            throw std::runtime_error("KV disk object id space is exhausted");
        }
        const std::uint64_t limit = base + kReservation;
        // Reserve before exposing an id to a write job.  A crash or failed
        // append can therefore create gaps but can never reuse an id whose
        // bytes might have reached a pack extent.
        persist_packset(limit);
        object_id_reservation_limit_ = limit;
    }
    const std::uint64_t id = next_object_id_++;
    ObjectRef ref;
    ref.kind      = kind;
    ref.bytes     = bytes;
    ref.live_refs = 0;
    ref.skip_refs = 0;
    objects_[id] = ref;
    return id;
}

void KVDiskCache::maybe_payload_io_stall() const {
    payload_io_entered_.store(true, std::memory_order_release);
    const int ms = payload_io_stall_ms_.load(std::memory_order_acquire);
    if (ms > 0) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
}

void KVDiskCache::finish_payload_io(std::unique_lock<std::mutex>& lock) {
    if (!lock.owns_lock()) { lock.lock(); }
    if (payload_io_inflight_ > 0) { --payload_io_inflight_; }
    if (spill_ && spill_->cancelled && !spill_->emergency && !spill_->committed &&
        payload_io_inflight_ == 0) {
        drop_spill(*spill_, lock);
        spill_.reset();
    }
    idle_cv_.notify_all();
    cv_.notify_all();
}

void KVDiskCache::write_new_object(DiskObjectKind kind, std::uint64_t id, const void* data,
                                   std::uint64_t bytes, std::unique_lock<std::mutex>* lock) {
    if ((data == nullptr && bytes != 0) || bytes == 0) {
        throw std::logic_error("KV disk packed object source is invalid");
    }
    const std::uint64_t extent = (bytes + kDiskPageIoAlignment - 1) &
                                 ~(static_cast<std::uint64_t>(kDiskPageIoAlignment) - 1);
    std::array<std::uint8_t, kDiskPageIoAlignment> padding{};
    std::pair<const void*, std::uint64_t> parts[2] = {{data, bytes}, {padding.data(), extent - bytes}};
    const std::size_t count = extent == bytes ? 1 : 2;
    write_new_object_parts(kind, id,
                           std::span<const std::pair<const void*, std::uint64_t>>(parts, count),
                           extent, lock);
    const auto it = objects_.find(id);
    if (it != objects_.end()) {
        it->second.location.extent_bytes = extent;
        it->second.location.stored_bytes = bytes;
        it->second.location.logical_bytes = bytes;
        it->second.location.record_crc32c = crc32c(
            std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(data),
                                           static_cast<std::size_t>(bytes)));
    }
}

void KVDiskCache::write_new_object_parts(DiskObjectKind kind, std::uint64_t id,
                                         std::span<const std::pair<const void*, std::uint64_t>> parts,
                                         std::uint64_t bytes, std::unique_lock<std::mutex>* lock) {
    if (bytes == 0 || bytes % kDiskPageIoAlignment != 0) {
        throw std::logic_error("KV disk pack extent is not 4 KiB aligned");
    }
    std::vector<iovec> iov;
    iov.reserve(parts.size());
    std::uint64_t total = 0;
    for (const auto& part : parts) {
        if (part.first == nullptr || part.second == 0) { continue; }
        if (part.second > std::numeric_limits<std::size_t>::max() - total) {
            throw std::logic_error("KV disk pack record is too large");
        }
        iov.push_back({const_cast<void*>(part.first), static_cast<std::size_t>(part.second)});
        total += part.second;
    }
    if (total != bytes || iov.empty() || iov.size() > IOV_MAX) {
        throw std::logic_error("KV disk pack record iovec shape is invalid");
    }
    if (total > static_cast<std::uint64_t>(SSIZE_MAX)) {
        throw std::logic_error("KV disk pack record exceeds write size limit");
    }
    const std::size_t index = kind_index(kind);
    std::uint32_t segment = pack_active_segment_[index];
    std::uint64_t offset = pack_active_tail_[index];
    if (offset + bytes < offset) { throw std::runtime_error("KV disk pack offset overflow"); }
    if (offset != 0 && offset + bytes > kPackSegmentBytes) {
        segment = ++pack_active_segment_[index];
        offset = 0;
        std::filesystem::create_directories(pack_path(kind, segment).parent_path());
        const int made = ::open(pack_path(kind, segment).c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
        if (made < 0) { throw std::runtime_error("failed to create KV disk pack rollover segment"); }
        ::close(made);
    }
    if (bytes > kPackSegmentBytes || offset > kPackSegmentBytes - bytes) {
        throw std::runtime_error("KV disk pack record exceeds segment capacity");
    }
    // Reserve the append range before releasing the cache lock.  A partial
    // write is permanent pack garbage and must never be overwritten by a
    // later writer in this process.
    pack_active_segment_[index] = segment;
    pack_active_tail_[index] = offset + bytes;
    if (lock != nullptr) { lock->unlock(); }
    try {
        maybe_payload_io_stall();
        const auto descriptor = acquire_pack_descriptor(active_generation_, kind, segment, false);
        const int fd = descriptor->buffered;
        if (fail_object_write_) {
            fail_object_write_ = false;
            throw std::runtime_error("injected KV disk object write failure");
        }
        std::uint64_t done = 0;
        std::size_t first = 0;
        while (done < total) {
            const ssize_t n = pwritev_with_test_limit(
                fd, iov.data() + first, static_cast<int>(iov.size() - first),
                static_cast<off_t>(offset + done), take_test_partial_pwritev_bytes());
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) { throw std::runtime_error("KV disk pack pwrite failed"); }
            done += static_cast<std::uint64_t>(n);
            ssize_t left = n;
            while (left > 0 && first < iov.size()) {
                auto& cur = iov[first];
                if (static_cast<std::size_t>(left) < cur.iov_len) {
                    cur.iov_base = static_cast<char*>(cur.iov_base) + left;
                    cur.iov_len -= static_cast<std::size_t>(left);
                    break;
                }
                left -= static_cast<ssize_t>(cur.iov_len);
                ++first;
            }
        }
        maybe_test_fault(DiskFaultPoint::AfterRecordWrite);
    } catch (...) {
        if (lock != nullptr && !lock->owns_lock()) { lock->lock(); }
        objects_.erase(id);
        throw;
    }
    if (lock != nullptr && !lock->owns_lock()) { lock->lock(); }
    auto it = objects_.find(id);
    if (it != objects_.end()) {
        it->second.location.generation = active_generation_;
        it->second.bytes = bytes;
        it->second.location.segment = segment;
        it->second.location.offset = offset;
        it->second.location.stored_bytes = bytes;
    }
}

std::uint64_t KVDiskCache::write_page_object(DiskObjectKind kind, const void* payload,
                                             std::size_t bytes,
                                             std::unique_lock<std::mutex>& lock) {
    std::uint8_t hdr[kDiskPageHeaderBytes]{};
    std::memcpy(hdr, kDiskPageMagic, 8);
    const std::uint32_t version = kDiskFormatVersion;
    std::memcpy(hdr + 8, &version, 4);
    const std::uint32_t payload_crc = crc32c(std::span<const std::uint8_t>(
        static_cast<const std::uint8_t*>(payload), bytes));
    std::memcpy(hdr + 12, &payload_crc, 4);
    std::pair<const void*, std::uint64_t> parts[3] = {
        {hdr, kDiskPageHeaderBytes},
        {payload, bytes},
    };
    int nparts = (payload != nullptr && bytes != 0) ? 2 : 1;
    const std::uint64_t file_bytes = aligned_page_file_bytes(bytes);
    const std::size_t pad_bytes =
        static_cast<std::size_t>(file_bytes) - kDiskPageHeaderBytes - bytes;
    std::array<std::uint8_t, kDiskPageIoAlignment> padding{};
    if (pad_bytes != 0) { parts[nparts++] = {padding.data(), pad_bytes}; }
    const std::uint64_t id        = allocate_object(kind, file_bytes);
    write_new_object_parts(kind, id, std::span<const std::pair<const void*, std::uint64_t>>(parts, nparts),
                           file_bytes, &lock);
    const auto it = objects_.find(id);
    if (it != objects_.end()) {
        it->second.location.extent_bytes = file_bytes;
        it->second.location.stored_bytes = kDiskPageHeaderBytes + bytes;
        it->second.location.logical_bytes = bytes;
        it->second.location.record_crc32c =
            crc32c_combine(crc32c(std::span<const std::uint8_t>(hdr, sizeof(hdr))),
                           payload_crc, bytes);
    }
    return id;
}

KVDiskCache::EncodedStateBlob KVDiskCache::encode_state_blob(
    DiskStateKind kind, const void* a, std::size_t na, const void* b, std::size_t nb) const {
    const std::size_t unc = na + nb;
    if (unc < na || (a == nullptr && na != 0) || (b == nullptr && nb != 0)) {
        throw std::logic_error("invalid KV disk state source");
    }
    EncodedStateBlob blob;
    blob.kind = kind;
    blob.first = a;
    blob.first_bytes = na;
    blob.second = b;
    blob.second_bytes = nb;
    if (config_.compress == KvDiskCompress::Zstd && unc != 0 && !force_zstd_fail_) {
        std::vector<std::uint8_t> joined;
        const void* src = a;
        std::size_t src_n = unc;
        if (nb != 0) {
            joined.resize(unc);
            if (na != 0) { std::memcpy(joined.data(), a, na); }
            if (nb != 0) { std::memcpy(joined.data() + na, b, nb); }
            src   = joined.data();
            src_n = unc;
        }
        blob.compressed.resize(ZSTD_compressBound(src_n));
        const std::size_t z =
            ZSTD_compress(blob.compressed.data(), blob.compressed.size(), src, src_n, 1);
        if (!ZSTD_isError(z)) {
            blob.codec = DiskCodec::Zstd;
            blob.compressed.resize(z);
        } else {
            blob.compressed.clear();
        }
    }
    return blob;
}

std::uint64_t KVDiskCache::write_state_blob(const EncodedStateBlob& blob,
                                             std::unique_lock<std::mutex>& lock) {
    const std::size_t unc = blob.uncompressed_bytes();
    if (unc == 0) { return 0; }
    const void* p0 = blob.codec == DiskCodec::Zstd ? blob.compressed.data() : blob.first;
    std::size_t n0 = blob.codec == DiskCodec::Zstd ? blob.compressed.size() : blob.first_bytes;
    const void* p1 = blob.codec == DiskCodec::Zstd ? nullptr : blob.second;
    std::size_t n1 = blob.codec == DiskCodec::Zstd ? 0 : blob.second_bytes;
    std::uint8_t hdr[kDiskCodecHeaderBytes]{};
    hdr[0] = static_cast<std::uint8_t>(blob.codec);
    hdr[1] = static_cast<std::uint8_t>(blob.kind);
    const std::uint64_t unc64 = unc;
    const std::uint64_t cmp64 = n0 + n1;
    std::memcpy(hdr + 4, &unc64, 8);
    std::memcpy(hdr + 12, &cmp64, 8);
    std::uint32_t state_crc = ~0U;
    if (n0 != 0) {
        state_crc = crc32c_update(state_crc, std::span<const std::uint8_t>(
            static_cast<const std::uint8_t*>(p0), n0));
    }
    if (n1 != 0) {
        state_crc = crc32c_update(state_crc, std::span<const std::uint8_t>(
            static_cast<const std::uint8_t*>(p1), n1));
    }
    state_crc = ~state_crc;
    std::memcpy(hdr + 20, &state_crc, 4);
    std::array<std::uint8_t, kDiskPageIoAlignment> padding{};
    std::pair<const void*, std::uint64_t> parts[5];
    int nparts = 0;
    parts[nparts++] = {hdr, kDiskCodecHeaderBytes};
    parts[nparts++] = {padding.data(), kDiskStatePayloadOffset - kDiskCodecHeaderBytes};
    if (p0 != nullptr && n0 != 0) { parts[nparts++] = {p0, n0}; }
    if (p1 != nullptr && n1 != 0) { parts[nparts++] = {p1, n1}; }
    const std::uint64_t file_bytes = aligned_state_file_bytes(n0 + n1);
    const std::size_t tail_padding =
        static_cast<std::size_t>(file_bytes) - kDiskStatePayloadOffset - n0 - n1;
    if (tail_padding != 0) { parts[nparts++] = {padding.data(), tail_padding}; }
    const std::uint64_t id        = allocate_object(DiskObjectKind::State, file_bytes);
    write_new_object_parts(DiskObjectKind::State, id,
                           std::span<const std::pair<const void*, std::uint64_t>>(parts, nparts),
                           file_bytes, &lock);
    const auto it = objects_.find(id);
    if (it != objects_.end()) {
        it->second.location.extent_bytes = file_bytes;
        it->second.location.stored_bytes = kDiskStatePayloadOffset + n0 + n1;
        it->second.location.logical_bytes = unc;
        it->second.location.record_crc32c = crc32c_combine(
            crc32c_parts(std::span<const std::pair<const void*, std::uint64_t>>(parts, 2)),
            state_crc, n0 + n1);
    }
    return id;
}

bool KVDiskCache::read_object(DiskObjectKind kind, std::uint64_t id, void* dst, std::uint64_t bytes) {
    if (id == 0 || dst == nullptr) { return bytes == 0; }
    ObjectRef::Location location;
    {
        std::lock_guard lock(mutex_);
        const auto it = objects_.find(id);
        if (it == objects_.end() || it->second.kind != kind ||
            bytes > it->second.location.stored_bytes ||
            it->second.location.stored_bytes > it->second.location.extent_bytes) {
            return false;
        }
        location = it->second.location;
    }
    maybe_payload_io_stall();
    std::shared_ptr<PackDescriptor> descriptor;
    ScopedFd startup_descriptor;
    int fd = -1;
    if (startup_loading_index_) {
        const auto path = pack_path(*location.generation, kind, location.segment);
        startup_descriptor.value = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (startup_descriptor.value < 0) { return false; }
        fd = startup_descriptor.value;
    } else {
        try {
            descriptor = acquire_pack_descriptor(location.generation, kind, location.segment,
                                                 false);
        } catch (...) { return false; }
        fd = descriptor->buffered;
    }
    std::uint64_t off = 0;
    auto* p           = static_cast<char*>(dst);
    while (off < bytes) {
        const ssize_t n =
            ::pread(fd, p + off, static_cast<std::size_t>(bytes - off),
                    static_cast<off_t>(location.offset + off));
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) {
            return false;
        }
        off += static_cast<std::uint64_t>(n);
    }
    if (bytes == location.stored_bytes &&
        crc32c(std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(dst),
                                               static_cast<std::size_t>(bytes))) !=
            location.record_crc32c) {
        return false;
    }
    return true;
}

bool KVDiskCache::read_page_payload(DiskObjectKind kind, std::uint64_t id, void* io, void* dst,
                                    std::size_t payload) {
    maybe_page_read_barrier();
    if (fail_next_page_read_.exchange(false, std::memory_order_acq_rel)) {
        throw std::runtime_error("injected KV disk page read failure");
    }
    if (io == nullptr || dst != static_cast<unsigned char*>(io) + kDiskPageHeaderBytes ||
        reinterpret_cast<std::uintptr_t>(io) % kDiskPageIoAlignment != 0) {
        return false;
    }
    ObjectRef::Location location;
    {
        std::lock_guard lock(mutex_);
        const auto it = objects_.find(id);
        if (it == objects_.end() || it->second.kind != kind ||
            it->second.location.extent_bytes == 0 ||
            it->second.location.stored_bytes < kDiskPageHeaderBytes ||
            it->second.location.logical_bytes != payload ||
            it->second.location.stored_bytes != kDiskPageHeaderBytes + payload ||
            it->second.location.extent_bytes != aligned_page_file_bytes(payload)) {
            return false;
        }
        location = it->second.location;
    }
    maybe_payload_io_stall();
    std::shared_ptr<PackDescriptor> descriptor;
    try { descriptor = acquire_pack_descriptor(location.generation, kind, location.segment, true); }
    catch (...) { return false; }
    const int fd = descriptor->direct;
    const std::size_t bytes = static_cast<std::size_t>(location.extent_bytes);
    std::size_t off = 0;
    while (off < bytes) {
        const ssize_t n = ::pread(fd, static_cast<unsigned char*>(io) + off, bytes - off,
                                  static_cast<off_t>(location.offset + off));
        if (n < 0 && errno == EINTR) { continue; }
        if (n <= 0) {
            return false;
        }
        off += static_cast<std::size_t>(n);
    }
    const auto* hdr = static_cast<const std::uint8_t*>(io);
    if (std::memcmp(hdr, kDiskPageMagic, 8) != 0) { return false; }
    std::uint32_t version = 0;
    std::memcpy(&version, hdr + 8, 4);
    std::uint32_t expected_crc = 0;
    std::memcpy(&expected_crc, hdr + 12, 4);
    const std::uint32_t payload_crc = crc32c(std::span<const std::uint8_t>(
        static_cast<const std::uint8_t*>(dst), payload));
    const std::uint32_t record_crc = crc32c_combine(
        crc32c(std::span<const std::uint8_t>(hdr, kDiskPageHeaderBytes)), payload_crc,
        payload);
    if (version != kDiskFormatVersion || expected_crc != payload_crc ||
        location.record_crc32c != record_crc) {
        return false;
    }
    return true;
}

bool KVDiskCache::decode_state(std::uint64_t id, DiskStateKind expected, void* dst,
                               std::size_t bytes) {
    return decode_state_parts(id, expected, dst, bytes, nullptr, 0);
}

bool KVDiskCache::decode_state_parts(std::uint64_t id, DiskStateKind expected, void* dst,
                                     std::size_t bytes, void* dst2, std::size_t bytes2,
                                     std::uint32_t read_workers) {
    maybe_state_decode_stall();
    const std::size_t total = bytes + bytes2;
    if (total < bytes) { return false; }
    if (id == 0) { return total == 0; }
    ObjectRef::Location location;
    {
        std::lock_guard lock(mutex_);
        const auto object = objects_.find(id);
        if (object == objects_.end() || object->second.kind != DiskObjectKind::State) {
            return false;
        }
        location = object->second.location;
    }
    maybe_payload_io_stall();
    std::shared_ptr<PackDescriptor> descriptor;
    try {
        descriptor = acquire_pack_descriptor(location.generation, DiskObjectKind::State,
                                              location.segment, false);
    }
    catch (...) { return false; }
    int fd = descriptor->buffered;
    struct stat st {};
    if (::fstat(fd, &st) != 0 || location.extent_bytes < kDiskStatePayloadOffset ||
        location.stored_bytes < kDiskStatePayloadOffset ||
        location.stored_bytes > location.extent_bytes ||
        static_cast<std::uint64_t>(st.st_size) < location.offset + location.extent_bytes) {
        return false;
    }
    const std::uint64_t file_bytes = location.extent_bytes;
    auto pread_fully = [&fd, &location](void* buf, std::uint64_t n, std::uint64_t off) {
        auto* p           = static_cast<char*>(buf);
        std::uint64_t got = 0;
        while (got < n) {
            const ssize_t r =
                ::pread(fd, p + got, static_cast<std::size_t>(n - got),
                        static_cast<off_t>(location.offset + off + got));
            if (r < 0 && errno == EINTR) { continue; }
            if (r <= 0) { return false; }
            got += static_cast<std::uint64_t>(r);
        }
        return true;
    };
    std::uint8_t hdr[kDiskCodecHeaderBytes];
    if (!pread_fully(hdr, kDiskCodecHeaderBytes, 0)) {
        return false;
    }
    std::array<std::uint8_t, kDiskStatePayloadOffset - kDiskCodecHeaderBytes> gap{};
    if (!pread_fully(gap.data(), gap.size(), kDiskCodecHeaderBytes)) {
        return false;
    }
    const auto codec = static_cast<DiskCodec>(hdr[0]);
    const auto kind  = static_cast<DiskStateKind>(hdr[1]);
    if (kind != expected) {
        return false;
    }
    std::uint64_t unc = 0;
    std::uint64_t cmp = 0;
    std::uint32_t expected_crc = 0;
    std::memcpy(&unc, hdr + 4, 8);
    std::memcpy(&cmp, hdr + 12, 8);
    std::memcpy(&expected_crc, hdr + 20, 4);
    if (unc != total) {
        return false;
    }
    if (!codec_payload_ok(codec, unc, cmp, total) ||
        !payload_fits(file_bytes, kDiskStatePayloadOffset, cmp) ||
        location.logical_bytes != unc ||
        location.stored_bytes != kDiskStatePayloadOffset + cmp ||
        cmp > std::numeric_limits<std::size_t>::max() - kDiskStatePayloadOffset) {
        return false;
    }
    if (codec == DiskCodec::Raw) {
        if (cmp != total || (bytes != 0 && dst == nullptr) || (bytes2 != 0 && dst2 == nullptr)) {
            return false;
        }
        const auto aligned = [](const void* p, std::size_t n) {
            return n % kDiskPageIoAlignment == 0 &&
                   (n == 0 || reinterpret_cast<std::uintptr_t>(p) % kDiskPageIoAlignment == 0);
        };
        const bool direct = aligned(dst, bytes) && aligned(dst2, bytes2);
        if (direct) {
            maybe_direct_state_read_barrier();
            try {
                descriptor = acquire_pack_descriptor(location.generation, DiskObjectKind::State,
                                                      location.segment, true);
            }
            catch (...) { return false; }
            fd = descriptor->direct;
        }
        std::uint32_t payload_crc = 0;
        if (direct && read_workers > 1 && total >= (8ULL << 20)) {
            constexpr std::size_t kChunkBytes = 4ULL << 20;
            struct Chunk {
                std::uint8_t* dst = nullptr;
                std::size_t bytes = 0;
                std::uint64_t payload_offset = 0;
                std::uint32_t crc = 0;
            };
            std::vector<Chunk> chunks;
            chunks.reserve((total + kChunkBytes - 1) / kChunkBytes + 1);
            std::uint64_t payload_offset = 0;
            auto append_chunks = [&](void* output, std::size_t output_bytes) {
                auto* p = static_cast<std::uint8_t*>(output);
                for (std::size_t off = 0; off < output_bytes; off += kChunkBytes) {
                    const std::size_t n = std::min(kChunkBytes, output_bytes - off);
                    chunks.push_back({p + off, n, payload_offset + off, 0});
                }
                payload_offset += output_bytes;
            };
            if (bytes != 0) { append_chunks(dst, bytes); }
            if (bytes2 != 0) { append_chunks(dst2, bytes2); }
            std::atomic<std::size_t> next{0};
            std::atomic<bool> failed{false};
            auto run = [&] {
                for (;;) {
                    const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= chunks.size() || failed.load(std::memory_order_relaxed)) { break; }
                    Chunk& chunk = chunks[index];
                    std::size_t got = 0;
                    while (got < chunk.bytes) {
                        const ssize_t n = ::pread(
                            fd, chunk.dst + got, chunk.bytes - got,
                            static_cast<off_t>(location.offset + kDiskStatePayloadOffset +
                                               chunk.payload_offset + got));
                        if (n < 0 && errno == EINTR) { continue; }
                        if (n <= 0) {
                            failed.store(true, std::memory_order_relaxed);
                            break;
                        }
                        got += static_cast<std::size_t>(n);
                    }
                    if (got == chunk.bytes) {
                        chunk.crc = crc32c(std::span<const std::uint8_t>(chunk.dst, chunk.bytes));
                    }
                }
            };
            const std::uint32_t workers = std::min<std::uint32_t>(
                std::max(1u, read_workers), static_cast<std::uint32_t>(chunks.size()));
            std::vector<std::thread> pool;
            pool.reserve(workers > 0 ? workers - 1 : 0);
            try {
                for (std::uint32_t i = 1; i < workers; ++i) { pool.emplace_back(run); }
                run();
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
            for (auto& thread : pool) {
                if (thread.joinable()) { thread.join(); }
            }
            if (failed.load(std::memory_order_relaxed)) { return false; }
            payload_crc = chunks.front().crc;
            for (std::size_t i = 1; i < chunks.size(); ++i) {
                payload_crc = crc32c_combine(payload_crc, chunks[i].crc, chunks[i].bytes);
            }
        } else {
            iovec iov[2] = {{dst, bytes}, {dst2, bytes2}};
            int first = bytes == 0 ? 1 : 0;
            std::size_t got = 0;
            while (got < total) {
                const ssize_t n = ::preadv(
                    fd, iov + first, 2 - first,
                    static_cast<off_t>(location.offset + kDiskStatePayloadOffset + got));
                if (n < 0 && errno == EINTR) { continue; }
                if (n <= 0) { return false; }
                got += static_cast<std::size_t>(n);
                ssize_t left = n;
                while (left > 0 && first < 2) {
                    auto& cur = iov[first];
                    if (static_cast<std::size_t>(left) < cur.iov_len) {
                        cur.iov_base = static_cast<char*>(cur.iov_base) + left;
                        cur.iov_len -= static_cast<std::size_t>(left);
                        break;
                    }
                    left -= static_cast<ssize_t>(cur.iov_len);
                    ++first;
                }
            }
            std::uint32_t actual_crc = ~0U;
            if (bytes != 0) {
                actual_crc = crc32c_update(actual_crc, std::span<const std::uint8_t>(
                    static_cast<const std::uint8_t*>(dst), bytes));
            }
            if (bytes2 != 0) {
                actual_crc = crc32c_update(actual_crc, std::span<const std::uint8_t>(
                    static_cast<const std::uint8_t*>(dst2), bytes2));
            }
            payload_crc = ~actual_crc;
        }
        if (expected_crc != payload_crc) { return false; }
        const std::pair<const void*, std::uint64_t> prefix_parts[2] = {
            {hdr, sizeof(hdr)}, {gap.data(), gap.size()}};
        return location.record_crc32c ==
               crc32c_combine(crc32c_parts(prefix_parts), payload_crc, cmp);
    }
    const std::size_t need = static_cast<std::size_t>(cmp);
    std::vector<std::uint8_t> blob;
    try {
        blob.resize(need);
    } catch (...) {
        return false;
    }
    if (need != 0 && !pread_fully(blob.data(), need, kDiskStatePayloadOffset)) {
        return false;
    }
    const std::uint32_t payload_crc = crc32c(blob);
    const std::pair<const void*, std::uint64_t> prefix_parts[2] = {
        {hdr, sizeof(hdr)}, {gap.data(), gap.size()}};
    if (expected_crc != payload_crc ||
        location.record_crc32c !=
            crc32c_combine(crc32c_parts(prefix_parts), payload_crc, cmp)) {
        return false;
    }
    const std::uint8_t* payload = blob.data();
    if (codec != DiskCodec::Zstd) { return false; }
    if (bytes2 == 0) {
        const std::size_t z = ZSTD_decompress(dst, bytes, payload, static_cast<std::size_t>(cmp));
        return !ZSTD_isError(z) && z == bytes;
    }
    std::vector<std::uint8_t> decoded(total);
    const std::size_t z = ZSTD_decompress(decoded.data(), total, payload,
                                          static_cast<std::size_t>(cmp));
    if (ZSTD_isError(z) || z != total) { return false; }
    if (bytes != 0) { std::memcpy(dst, decoded.data(), bytes); }
    if (bytes2 != 0) { std::memcpy(dst2, decoded.data() + bytes, bytes2); }
    return true;
}

bool KVDiskCache::decode_state_parallel(std::vector<StateDecodeJob>& jobs) {
    std::uint32_t n = 0;
    for (const auto& job : jobs) {
        if (job.id != 0 && ((job.bytes != 0 && job.dst != nullptr) ||
                            (job.bytes2 != 0 && job.dst2 != nullptr))) {
            ++n;
        }
    }
    if (n == 0) { return true; }
    auto run = [&](StateDecodeJob& job) {
        if (job.id == 0 || (job.bytes == 0 && job.bytes2 == 0) ||
            (job.bytes != 0 && job.dst == nullptr) ||
            (job.bytes2 != 0 && job.dst2 == nullptr)) {
            job.ok = job.id == 0 || (job.bytes == 0 && job.bytes2 == 0);
            return;
        }
        try {
            const std::uint32_t read_workers =
                std::min(4u, std::max(1u, restore_io_threads_ / n));
            job.ok = decode_state_parts(job.id, job.kind, job.dst, job.bytes, job.dst2,
                                        job.bytes2, read_workers);
        } catch (...) { job.ok = false; }
    };
    const std::uint32_t workers = std::min(restore_io_threads_, n);
    if (workers <= 1) {
        for (auto& job : jobs) { run(job); }
    } else {
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> pool;
        pool.reserve(workers);
        try {
            for (std::uint32_t i = 0; i < workers; ++i) {
                pool.emplace_back([&] {
                    for (;;) {
                        const std::size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= jobs.size()) { break; }
                        run(jobs[idx]);
                    }
                });
            }
        } catch (...) {
            for (auto& t : pool) {
                if (t.joinable()) { t.join(); }
            }
            return false;
        }
        for (auto& t : pool) { t.join(); }
    }
    for (const auto& job : jobs) {
        if (job.id != 0 && (job.bytes != 0 || job.bytes2 != 0) && !job.ok) { return false; }
    }
    return true;
}

std::optional<DiskMatch> KVDiskCache::plan_match(const PreparedPromptData& prompt,
                                                 std::span<const PrefixHash128> hash_chain) {
    std::lock_guard lock(mutex_);
    std::optional<DiskMatch> best;
    for (std::uint64_t id : fifo_) {
        const IndexEntry& record = require(id);
        if (record.pinned) { continue; }
        const DiskMeta& meta = record.meta;
        const auto hash_hits = [&](std::uint32_t frontier, PrefixHash128 hash) {
            return frontier > 0 && frontier < hash_chain.size() && hash_chain[frontier] == hash;
        };
        const bool frontier_hash = hash_hits(meta.execution_frontier, meta.hash_f) &&
                                    !checkpoint_missing_hidden(meta.current_hidden_id,
                                                             config_.hidden_bytes);
        const bool checkpoint_hash =
            meta.hash_c_valid && hash_hits(meta.rewrite_frontier, meta.hash_c) &&
            !checkpoint_missing_hidden(meta.rewrite_hidden_id, config_.hidden_bytes);
        bool ladder_hash = false;
        auto consider_slot = [&](const DiskCheckpointSlot& slot) {
            if (slot.frontier != 0 && hash_hits(slot.frontier, slot.hash) &&
                !checkpoint_missing_hidden(slot.hidden_id, config_.hidden_bytes)) {
                ladder_hash = true;
            }
        };
        consider_slot(meta.rollback);
        consider_slot(meta.ladders[0]);
        consider_slot(meta.ladders[1]);
        if (!frontier_hash && !checkpoint_hash && !ladder_hash) { continue; }

        DiskMatch candidate;
        candidate.entry_id            = id;
        candidate.hash_f              = meta.hash_f;
        candidate.execution_frontier  = meta.execution_frontier;
        const auto consider = [&](PrefixReusePath path, std::uint32_t base) {
            if (base == 0) { return; }
            ++exact_comparisons_;
            if (!prefix_matches(prompt, record.ledger, record.identity, base)) { return; }
            if (base > candidate.reuse_base) {
                candidate.reuse      = path;
                candidate.reuse_base = base;
            }
        };
        if (frontier_hash) { consider(PrefixReusePath::AppendAtFrontier, meta.execution_frontier); }
        if (checkpoint_hash) { consider(rewrite_path(meta.rewrite_kind), meta.rewrite_frontier); }
        auto consider_saved = [&](const DiskCheckpointSlot& slot) {
            if (slot.frontier != 0 && hash_hits(slot.frontier, slot.hash) &&
                !checkpoint_missing_hidden(slot.hidden_id, config_.hidden_bytes)) {
                consider(reuse_path_for_context_checkpoint_kind(slot.kind), slot.frontier);
            }
        };
        consider_saved(meta.rollback);
        consider_saved(meta.ladders[0]);
        consider_saved(meta.ladders[1]);
        if (candidate.reuse_base == 0) { continue; }
        if (!best || candidate.reuse_base > best->reuse_base) { best = candidate; }
    }
    return best;
}

std::optional<DiskRestoredHost> KVDiskCache::load_host(std::uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(entry_id);
    if (it == entries_.end()) { return std::nullopt; }
    const IndexEntry& record = it->second;
    const DiskMeta& meta     = record.meta;
    DiskRestoredHost out;
    out.execution_frontier      = meta.execution_frontier;
    out.ledger_frontier         = meta.ledger_frontier;
    out.rope_delta              = meta.rope_delta;
    out.text_kv_valid           = meta.text_kv_valid;
    out.mtp_kv_valid            = meta.mtp_kv_valid;
    out.dflash_context_frontier = meta.dflash_context_frontier;
    out.tail_hidden_valid       = meta.tail_hidden_valid &&
                            !checkpoint_missing_hidden(meta.current_hidden_id, config_.hidden_bytes);
    out.rewrite_valid           = meta.rewrite_valid &&
                            !checkpoint_missing_hidden(meta.rewrite_hidden_id, config_.hidden_bytes);
    out.rewrite_kind            = meta.rewrite_kind;
    out.rewrite_frontier        = meta.rewrite_frontier;
    out.backend_image_present   = !meta.backend_page_ids.empty();
    out.ledger                  = record.ledger;
    out.identity                = record.identity;
    out.disk_entry_id           = entry_id;
    auto push_slot = [&](const DiskCheckpointSlot& slot) {
        if (slot.frontier == 0) { return; }
        if (checkpoint_missing_hidden(slot.hidden_id, config_.hidden_bytes)) { return; }
        RamLadderIndex index;
        index.frontier = slot.frontier;
        index.hash     = slot.hash;
        index.kind     = slot.kind;
        out.ladders.push_back(index);
    };
    push_slot(meta.rollback);
    push_slot(meta.ladders[0]);
    push_slot(meta.ladders[1]);
    return out;
}

bool KVDiskCache::enqueue_unique_decode(
    std::unordered_map<std::uint64_t, DecodedCheckpointState>& decoded,
    std::vector<StateDecodeJob>& jobs, std::uint64_t id, DiskStateKind kind,
    std::size_t first_bytes, std::size_t second_bytes) {
    if (id == 0 || first_bytes + second_bytes == 0) { return true; }
    const auto it = decoded.find(id);
    if (it != decoded.end()) {
        const auto matches = [](const std::shared_ptr<PinnedHostBuffer>& buffer,
                                std::size_t bytes) {
            return bytes == 0 ? !buffer : buffer && buffer->size() == bytes;
        };
        return matches(it->second.first, first_bytes) &&
               matches(it->second.second, second_bytes);
    }
    auto& dst = decoded[id];
    try {
        if (first_bytes != 0) {
            dst.first = std::make_shared<PinnedHostBuffer>(first_bytes, kDiskPageIoAlignment);
        }
        if (second_bytes != 0) {
            dst.second = std::make_shared<PinnedHostBuffer>(second_bytes, kDiskPageIoAlignment);
        }
    } catch (...) { return false; }
    jobs.push_back(StateDecodeJob{id, kind,
                                  dst.first ? dst.first->data() : nullptr, first_bytes, true,
                                  dst.second ? dst.second->data() : nullptr, second_bytes});
    return true;
}

bool KVDiskCache::collect_checkpoint_decode_jobs(
    const DiskMeta& meta,
    std::unordered_map<std::uint64_t, DecodedCheckpointState>& decoded,
    std::vector<StateDecodeJob>& jobs, bool skip[3], std::size_t hid_n[3]) {
    const std::size_t conv_n = static_cast<std::size_t>(config_.fingerprint.gdn_conv_bytes);
    const std::size_t rec_n  = static_cast<std::size_t>(config_.fingerprint.gdn_recurrent_bytes);
    const std::size_t cyc_n  = static_cast<std::size_t>(config_.fingerprint.cyclic_lane_bytes);
    const DiskCheckpointSlot* const slot_ptrs[3] = {&meta.rollback, &meta.ladders[0],
                                                      &meta.ladders[1]};
    auto hidden_size = [&](std::uint64_t hidden_id) -> std::size_t {
        if (config_.hidden_bytes != 0) { return static_cast<std::size_t>(config_.hidden_bytes); }
        std::uint8_t hdr[kDiskCodecHeaderBytes]{};
        if (!read_object(DiskObjectKind::State, hidden_id, hdr, kDiskCodecHeaderBytes)) {
            return 0;
        }
        std::uint64_t unc = 0;
        std::memcpy(&unc, hdr + 4, 8);
        return static_cast<std::size_t>(unc);
    };
    for (int i = 0; i < 3; ++i) {
        skip[i]  = false;
        hid_n[i] = 0;
        const DiskCheckpointSlot& slot = *slot_ptrs[i];
        if (slot.frontier == 0 ||
            checkpoint_missing_hidden(slot.hidden_id, config_.hidden_bytes)) {
            skip[i] = true;
            continue;
        }
        const DiskStateKind gdn_kind = slot.kind == ContextCheckpointKind::TurnRollback
                                            ? DiskStateKind::RollbackGdn
                                            : DiskStateKind::LadderGdn;
        const DiskStateKind hid_kind = slot.kind == ContextCheckpointKind::TurnRollback
                                            ? DiskStateKind::RollbackHidden
                                            : DiskStateKind::LadderHidden;
        const DiskStateKind cyc_kind = slot.kind == ContextCheckpointKind::TurnRollback
                                            ? DiskStateKind::DflashRollback
                                            : DiskStateKind::DflashLadder;
        if (conv_n + rec_n != 0) {
            if (slot.gdn_id == 0) { return false; }
            if (!enqueue_unique_decode(decoded, jobs, slot.gdn_id, gdn_kind, conv_n, rec_n)) {
                return false;
            }
        }
        if (slot.hidden_id != 0) {
            const std::size_t n = hidden_size(slot.hidden_id);
            if (n == 0) { return false; }
            hid_n[i] = n;
            if (!enqueue_unique_decode(decoded, jobs, slot.hidden_id, hid_kind, n)) {
                return false;
            }
        } else if (config_.hidden_bytes != 0) {
            return false;
        }
        if (cyc_n != 0) {
            if (slot.dflash_id == 0) { return false; }
            if (!enqueue_unique_decode(decoded, jobs, slot.dflash_id, cyc_kind, cyc_n)) {
                return false;
            }
        }
    }
    return true;
}

bool KVDiskCache::assemble_checkpoint_images(
    DiskRestoredHost& host, const DiskMeta& meta,
    std::unordered_map<std::uint64_t, DecodedCheckpointState>& decoded, const bool skip[3],
    const std::size_t hid_n[3]) {
    const std::size_t conv_n = static_cast<std::size_t>(config_.fingerprint.gdn_conv_bytes);
    const std::size_t rec_n  = static_cast<std::size_t>(config_.fingerprint.gdn_recurrent_bytes);
    const std::size_t cyc_n  = static_cast<std::size_t>(config_.fingerprint.cyclic_lane_bytes);
    const DiskCheckpointSlot* const slot_ptrs[3] = {&meta.rollback, &meta.ladders[0],
                                                      &meta.ladders[1]};
    std::vector<DiskLadderImage> images;
    auto find = [&](std::uint64_t id) -> DecodedCheckpointState* {
        const auto it = decoded.find(id);
        return it == decoded.end() ? nullptr : &it->second;
    };
    for (int i = 0; i < 3; ++i) {
        if (skip[i]) { continue; }
        const DiskCheckpointSlot& slot = *slot_ptrs[i];
        DiskLadderImage image;
        image.frontier = slot.frontier;
        image.hash     = slot.hash;
        image.kind     = slot.kind;
        if (conv_n + rec_n != 0) {
            DecodedCheckpointState* state = find(slot.gdn_id);
            if (state == nullptr || (conv_n != 0 && !state->first) ||
                (rec_n != 0 && !state->second)) {
                return false;
            }
            image.conv      = state->first;
            image.recurrent = state->second;
        }
        if (slot.hidden_id != 0) {
            DecodedCheckpointState* state = find(slot.hidden_id);
            if (state == nullptr || !state->first || state->first->size() != hid_n[i]) {
                return false;
            }
            image.hidden = state->first;
        }
        if (cyc_n != 0) {
            DecodedCheckpointState* state = find(slot.dflash_id);
            if (state == nullptr || !state->first || state->first->size() != cyc_n) {
                return false;
            }
            image.dflash = state->first;
        }
        images.push_back(image);
    }
    host.ladder_images = std::move(images);
    return true;
}

bool KVDiskCache::populate_checkpoint_images(DiskRestoredHost& host) {
    return populate_checkpoint_images(host, 0, 0, 0, 0, PrefixReusePath::FullReset, 0, false);
}

bool KVDiskCache::populate_checkpoint_images(DiskRestoredHost& host, std::uint64_t epoch,
                                              std::uint64_t entry, PrefixReusePath reuse,
                                              std::uint32_t reuse_base, bool check_live) {
    return populate_checkpoint_images(host, epoch, entry, 0, 0, reuse, reuse_base, check_live);
}

bool KVDiskCache::populate_checkpoint_images(DiskRestoredHost& host, std::uint64_t epoch,
                                              std::uint64_t entry,
                                              std::uint64_t committed_generation,
                                              std::uint64_t claim_generation,
                                              PrefixReusePath reuse, std::uint32_t reuse_base,
                                              bool check_live) {
    CheckpointDecodePlan plan;
    if (!prepare_checkpoint_decode(host, plan, epoch, entry, committed_generation,
                                   claim_generation, reuse, reuse_base, check_live)) {
        return false;
    }
    return finish_checkpoint_decode(host, plan, epoch, entry, committed_generation,
                                    claim_generation, reuse, reuse_base, check_live);
}

bool KVDiskCache::prepare_checkpoint_decode(
    DiskRestoredHost& host, CheckpointDecodePlan& plan, std::uint64_t epoch,
    std::uint64_t entry, std::uint64_t committed_generation,
    std::uint64_t claim_generation, PrefixReusePath reuse, std::uint32_t reuse_base,
    bool check_live) {
    {
        std::lock_guard lock(mutex_);
        if (check_live &&
            !restore_state_is_live(epoch, entry, committed_generation, claim_generation, reuse,
                                    reuse_base)) {
            return false;
        }
        const auto it = entries_.find(host.disk_entry_id);
        if (it == entries_.end()) { return false; }
        plan.meta = it->second.meta;
    }
    host.ladder_images.clear();
    plan.decoded.clear();
    plan.jobs.clear();
    plan.jobs.reserve(9);
    if (!collect_checkpoint_decode_jobs(plan.meta, plan.decoded, plan.jobs, plan.skip,
                                        plan.hidden_bytes)) {
        return false;
    }
    if (!check_live) { return true; }
    if (checkpoint_prepare_barrier_armed_.load(std::memory_order_acquire)) {
        checkpoint_prepare_barrier_entered_.store(true, std::memory_order_release);
        while (checkpoint_prepare_barrier_armed_.load(std::memory_order_acquire) &&
               !checkpoint_prepare_barrier_continue_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
    std::lock_guard lock(mutex_);
    return restore_state_is_live(epoch, entry, committed_generation, claim_generation, reuse,
                                 reuse_base);
}

bool KVDiskCache::finish_checkpoint_decode(
    DiskRestoredHost& host, CheckpointDecodePlan& plan, std::uint64_t epoch,
    std::uint64_t entry, std::uint64_t committed_generation,
    std::uint64_t claim_generation, PrefixReusePath reuse, std::uint32_t reuse_base,
    bool check_live) {
    const auto still_live = [&] {
        if (!check_live) { return true; }
        std::lock_guard lock(mutex_);
        return restore_state_is_live(epoch, entry, committed_generation, claim_generation, reuse,
                                      reuse_base);
    };
    if (!still_live()) { return false; }
    if (!decode_state_parallel(plan.jobs)) { return false; }
    if (!still_live()) { return false; }
    return assemble_checkpoint_images(host, plan.meta, plan.decoded, plan.skip,
                                      plan.hidden_bytes);
}

bool KVDiskCache::claim(std::uint64_t entry_id) {
    return claim(entry_id, PrefixHash128{}, 0);
}

bool KVDiskCache::claim(std::uint64_t entry_id, PrefixHash128 expected_hash_f,
                        std::uint32_t expected_frontier, std::uint32_t expected_reuse_base,
                        PrefixReusePath expected_reuse) {
    std::unique_lock lock(mutex_);
    auto it = entries_.find(entry_id);
    if (it == entries_.end()) { return false; }
    if (it->second.pinned) { throw std::logic_error("disk cache entry is already claimed"); }
    const std::uint64_t hidden_bytes = config_.hidden_bytes;
    const auto hidden_ok = [hidden_bytes](std::uint64_t hidden_id) {
        return !checkpoint_missing_hidden(hidden_id, hidden_bytes);
    };
    const auto has_restore_head = [hidden_ok](const DiskMeta& meta, std::uint32_t base) {
        if (base == 0) { return true; }
        if (meta.execution_frontier == base && hidden_ok(meta.current_hidden_id)) { return true; }
        if (meta.rewrite_valid && meta.rewrite_frontier == base &&
            hidden_ok(meta.rewrite_hidden_id)) {
            return true;
        }
        if (meta.rollback.frontier == base && hidden_ok(meta.rollback.hidden_id)) { return true; }
        return (meta.ladders[0].frontier == base && hidden_ok(meta.ladders[0].hidden_id)) ||
               (meta.ladders[1].frontier == base && hidden_ok(meta.ladders[1].hidden_id));
    };
    const auto has_planned_head = [hidden_ok](const DiskMeta& meta, PrefixReusePath path,
                                                std::uint32_t base) {
        if (base == 0 || path == PrefixReusePath::FullReset) { return true; }
        switch (path) {
        case PrefixReusePath::AppendAtFrontier:
            return meta.execution_frontier == base && hidden_ok(meta.current_hidden_id);
        case PrefixReusePath::RestoreTurnCheckpoint:
            return meta.rewrite_valid && meta.rewrite_frontier == base &&
                   meta.rewrite_kind == RewriteCheckpointKind::TurnClosure &&
                   hidden_ok(meta.rewrite_hidden_id);
        case PrefixReusePath::RestoreResponseCheckpoint:
            return meta.rewrite_valid && meta.rewrite_frontier == base &&
                   meta.rewrite_kind == RewriteCheckpointKind::ResponseReplay &&
                   hidden_ok(meta.rewrite_hidden_id);
        case PrefixReusePath::RestoreTurnRollback:
            return meta.rollback.frontier == base &&
                   meta.rollback.kind == ContextCheckpointKind::TurnRollback &&
                   hidden_ok(meta.rollback.hidden_id);
        case PrefixReusePath::RestoreContextCheckpoint:
            return (meta.ladders[0].frontier == base &&
                    meta.ladders[0].kind == ContextCheckpointKind::Ladder &&
                    hidden_ok(meta.ladders[0].hidden_id)) ||
                   (meta.ladders[1].frontier == base &&
                    meta.ladders[1].kind == ContextCheckpointKind::Ladder &&
                    hidden_ok(meta.ladders[1].hidden_id));
        default:
            return false;
        }
    };
    const auto generation_ok = [&](const IndexEntry& record) {
        if (expected_frontier != 0 || expected_hash_f != PrefixHash128{}) {
            if (record.meta.hash_f != expected_hash_f ||
                record.meta.execution_frontier != expected_frontier) {
                return false;
            }
        }
        if (expected_reuse != PrefixReusePath::FullReset) {
            return has_planned_head(record.meta, expected_reuse, expected_reuse_base);
        }
        return has_restore_head(record.meta, expected_reuse_base);
    };
    if (!generation_ok(it->second)) { return false; }
    it->second.pinned = true;
    ++it->second.claim_generation;
    bump_version();
    if (spill_ && idle_rewrite_of(entry_id)) {
        spill_->cancelled = true;
        if (payload_io_inflight_ == 0) {
            drop_spill(*spill_, lock);
            spill_.reset();
        }
        idle_cv_.notify_all();
        cv_.notify_all();
    }
    idle_cv_.wait(lock, [&] { return !idle_rewrite_of(entry_id); });
    it = entries_.find(entry_id);
    if (it == entries_.end() || !it->second.pinned) { return false; }
    if (!generation_ok(it->second)) {
        it->second.pinned = false;
        bump_version();
        return false;
    }
    return true;
}

void KVDiskCache::release(std::uint64_t entry_id) {
    std::lock_guard lock(mutex_);
    IndexEntry& record = require(entry_id);
    if (!record.pinned) { throw std::logic_error("disk cache entry is not claimed"); }
    record.pinned = false;
    ++record.claim_generation;
    purge_prefetch_of(entry_id);
    bump_version();
}

void KVDiskCache::consume(std::uint64_t entry_id) {
    std::lock_guard lock(mutex_);
    IndexEntry& record = require(entry_id);
    if (!record.pinned) { throw std::logic_error("disk cache consume requires a claimed entry"); }
    record.pinned = false;
    ++record.claim_generation;
    purge_prefetch_of(entry_id);
    ++restores_;
    bump_version();
}

void KVDiskCache::note_ram_resident(std::uint64_t ram_id, std::uint64_t disk_ticket) {
    std::lock_guard lock(mutex_);
    RamNote& note          = ram_notes_[ram_id];
    note.ticket            = disk_ticket;
    note.durable           = false;
    note.failed_this_generation = false;
    note.generation_stamp  = durable_generation_;
}

void KVDiskCache::forget_ram_resident(std::uint64_t ram_id) noexcept {
    std::lock_guard lock(mutex_);
    ram_notes_.erase(ram_id);
}

bool KVDiskCache::ram_is_durable(std::uint64_t ram_id) const {
    std::lock_guard lock(mutex_);
    const auto it = ram_notes_.find(ram_id);
    return it != ram_notes_.end() && it->second.durable;
}

void KVDiskCache::request_idle_spill() {
    std::lock_guard lock(mutex_);
    idle_requested_ = true;
    cv_.notify_all();
}

void KVDiskCache::cancel_idle_locked(std::unique_lock<std::mutex>& lock) {
    ++idle_cancel_epoch_;
    idle_requested_ = false;
    idle_q_.clear();
    if (spill_ && !spill_->emergency) {
        spill_->cancelled = true;
        if (payload_io_inflight_ == 0) {
            drop_spill(*spill_, lock);
            spill_.reset();
        }
        idle_cv_.notify_all();
    }
}

void KVDiskCache::cancel_idle_spill() {
    std::unique_lock lock(mutex_);
    ++idle_cancel_epoch_;
    idle_cancel_all_ = true;
    cancel_idle_locked(lock);
    idle_cv_.wait(lock, [&] {
        return payload_io_inflight_ == 0 && (!spill_ || spill_->emergency) && !idle_pinning_;
    });
    if (spill_ && spill_->cancelled && !spill_->emergency) {
        drop_spill(*spill_, lock);
        spill_.reset();
        idle_cv_.notify_all();
    }
    idle_cancel_all_ = false;
}

void KVDiskCache::begin_ram_idle_exclusion(std::uint64_t ram_id) {
    std::unique_lock lock(mutex_);
    idle_cancel_rams_.insert(ram_id);
    if (spill_ && !spill_->emergency && spill_->ram_id == ram_id) {
        spill_->cancelled = true;
        if (payload_io_inflight_ == 0) {
            drop_spill(*spill_, lock);
            spill_.reset();
        }
        idle_cv_.notify_all();
        cv_.notify_all();
    }
    idle_q_.erase(std::remove_if(idle_q_.begin(), idle_q_.end(),
                                 [ram_id](const Job& job) { return job.ram_entry_id == ram_id; }),
                  idle_q_.end());
    idle_cv_.wait(lock, [&] {
        return payload_io_inflight_ == 0 &&
               (!spill_ || spill_->emergency || spill_->ram_id != ram_id) &&
               !(idle_pinning_ && idle_pinning_ram_ == ram_id);
    });
    if (spill_ && spill_->cancelled && !spill_->emergency && spill_->ram_id == ram_id) {
        drop_spill(*spill_, lock);
        spill_.reset();
        idle_cv_.notify_all();
    }
}

void KVDiskCache::end_ram_idle_exclusion(std::uint64_t ram_id) noexcept {
    std::lock_guard lock(mutex_);
    idle_cancel_rams_.erase(ram_id);
}

void KVDiskCache::cancel_idle_of_ram(std::uint64_t ram_id) {
    begin_ram_idle_exclusion(ram_id);
    end_ram_idle_exclusion(ram_id);
}

std::uint32_t KVDiskCache::share_pages(SpeculativeBackend backend, bool main,
                                       std::uint32_t share_tokens) const {
    if (main) { return ninfer::pages_for_tokens(share_tokens); }
    if (config_.backend_pool == nullptr) { return 0; }
    if (backend == SpeculativeBackend::Mtp) {
        return ninfer::pages_for_tokens(share_tokens == 0 ? 0U : share_tokens - 1U);
    }
    if (backend == SpeculativeBackend::DFlash) { return ninfer::pages_for_tokens(share_tokens); }
    return 0;
}

void KVDiskCache::enqueue(Job job, std::unique_lock<std::mutex>* /*lock*/) {
    switch (job.kind) {
    case JobKind::RestoreRead:
        restore_q_.push_back(job);
        break;
    case JobKind::EmergencySpillPage:
    case JobKind::EmergencyCommit:
        emergency_q_.push_back(job);
        break;
    case JobKind::IdleSpillPage:
    case JobKind::IdleCommit:
        idle_q_.push_back(job);
        break;
    case JobKind::PrefetchWindow:
        prefetch_q_.push_back(job);
        break;
    case JobKind::Stop:
        break;
    }
    cv_.notify_all();
}

std::optional<KVDiskCache::Job> KVDiskCache::take_job(
    std::unique_lock<std::mutex>& /*lock*/) {
    if (restore_io_threads_ == 1 && !restore_q_.empty() && !emergency_payload_busy_locked()) {
        Job job = restore_q_.front();
        restore_q_.pop_front();
        note_reader_claim_locked(job);
        return job;
    }
    if (!emergency_q_.empty() && !restore_readers_busy_locked()) {
        Job job = emergency_q_.front();
        emergency_q_.pop_front();
        ++payload_io_inflight_;
        return job;
    }
    if (!idle_q_.empty() && !restore_or_prefetch_busy_locked()) {
        Job job = idle_q_.front();
        idle_q_.pop_front();
        ++payload_io_inflight_;
        return job;
    }
    if (restore_io_threads_ == 1 && prefetch_readable_locked()) {
        Job job = prefetch_q_.front();
        prefetch_q_.pop_front();
        note_reader_claim_locked(job);
        return job;
    }
    return std::nullopt;
}

std::optional<KVDiskCache::Job> KVDiskCache::take_restore_job(
    std::unique_lock<std::mutex>& /*lock*/) {
    if (!restore_q_.empty() && !emergency_payload_busy_locked()) {
        Job job = restore_q_.front();
        restore_q_.pop_front();
        note_reader_claim_locked(job);
        return job;
    }
    if (prefetch_readable_locked()) {
        if (!idle_q_.empty()) {
            prefetch_preempted_idle_.fetch_add(1, std::memory_order_relaxed);
        }
        Job job = prefetch_q_.front();
        prefetch_q_.pop_front();
        note_reader_claim_locked(job);
        return job;
    }
    return std::nullopt;
}

bool KVDiskCache::prefetch_readable_locked() const noexcept {
    return !prefetch_q_.empty() && payload_io_inflight_ == 0 &&
           (stopping_ || (emergency_q_.empty() && !(spill_ && spill_->emergency) &&
                           idle_q_.empty()));
}

bool KVDiskCache::restore_readers_busy_locked() const noexcept {
    if (window_inflight_ != 0 || !restore_q_.empty() || !reader_claims_.empty()) {
        return true;
    }
    if (!stopping_ && spill_ && restore_active_ && restore_entry_id_ != 0 &&
        restore_entry_id_ == spill_->child_id) {
        for (const WindowSlot& slot : window_) {
            if (slot.assigned && slot.disk_entry_id == spill_->child_id) { return true; }
        }
    }
    return false;
}

bool KVDiskCache::restore_or_prefetch_busy_locked() const noexcept {
    return restore_active_ || restore_state_inflight_ > 0 || window_inflight_ != 0 ||
           !restore_q_.empty() || !reader_claims_.empty();
}

bool KVDiskCache::restore_cancels_idle_locked() const noexcept {
    return restore_active_ || restore_state_inflight_ > 0 || !restore_q_.empty();
}

bool KVDiskCache::emergency_payload_busy_locked() const noexcept {
    return payload_io_inflight_ > 0 && spill_ && spill_->emergency;
}

void KVDiskCache::note_reader_claim_locked(const Job& job) {
    reader_claims_.push_back({job.disk_entry_id, job.restore_epoch, job.claim_generation,
                                job.committed_generation, job.pool, job.logical_index});
}

void KVDiskCache::drop_reader_claim(const Job& job) noexcept {
    std::lock_guard lock(mutex_);
    for (auto it = reader_claims_.begin(); it != reader_claims_.end(); ++it) {
        if (it->disk_entry_id == job.disk_entry_id && it->restore_epoch == job.restore_epoch &&
            it->claim_generation == job.claim_generation &&
            it->committed_generation == job.committed_generation && it->pool == job.pool &&
            it->logical_index == job.logical_index) {
            reader_claims_.erase(it);
            break;
        }
    }
    idle_cv_.notify_all();
    cv_.notify_all();
}

bool KVDiskCache::job_matches_index(const Job& job) const noexcept {
    const auto it = entries_.find(job.disk_entry_id != 0 ? job.disk_entry_id : restore_entry_id_);
    if (it == entries_.end()) { return false; }
    return it->second.committed_generation == job.committed_generation &&
           it->second.claim_generation == job.claim_generation;
}

bool KVDiskCache::slot_matches_index(const WindowSlot& slot) const noexcept {
    if (slot.disk_entry_id == 0) { return false; }
    const auto it = entries_.find(slot.disk_entry_id);
    if (it == entries_.end()) { return false; }
    return it->second.committed_generation == slot.committed_generation &&
           it->second.claim_generation == slot.claim_generation;
}

bool KVDiskCache::page_job_is_live_restore(const Job& job) const noexcept {
    if (!restore_target_ || job.restore_epoch != restore_epoch_) { return false; }
    const std::uint64_t entry = job.disk_entry_id != 0 ? job.disk_entry_id : restore_entry_id_;
    if (entry != restore_entry_id_ || !job_matches_index(job)) { return false; }
    return job.kind == JobKind::RestoreRead || job.kind == JobKind::PrefetchWindow;
}

bool KVDiskCache::slot_is_live_restore_page(const WindowSlot& slot) const noexcept {
    return restore_target_.has_value() && slot.disk_entry_id == restore_entry_id_ &&
           slot.epoch == restore_epoch_ && slot_matches_index(slot);
}

bool KVDiskCache::restore_job_is_live(const Job& job) const noexcept {
    if (job.kind == JobKind::RestoreRead) {
        if (job.restore_epoch != restore_epoch_) { return false; }
        return restore_target_.has_value() && job.disk_entry_id == restore_entry_id_ &&
               job_matches_index(job);
    }
    if (job.kind == JobKind::PrefetchWindow) {
        if (job.restore_epoch != restore_epoch_) { return false; }
        if (restore_target_ && job.disk_entry_id != restore_entry_id_) { return false; }
        if (!job_matches_index(job)) { return false; }
        const auto it = entries_.find(job.disk_entry_id);
        if (it == entries_.end()) { return false; }
        if (restore_target_ && job.disk_entry_id == restore_entry_id_) { return true; }
        return it->second.pinned;
    }
    return job.restore_epoch == restore_epoch_;
}

void KVDiskCache::maybe_restore_job_barrier() {
    if (!restore_job_barrier_armed_.load(std::memory_order_acquire)) { return; }
    restore_job_dequeued_.store(true, std::memory_order_release);
    while (restore_job_barrier_armed_.load(std::memory_order_acquire) &&
           !restore_job_continue_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool KVDiskCache::restore_state_is_live(std::uint64_t epoch, std::uint64_t entry,
                                          PrefixReusePath reuse,
                                          std::uint32_t reuse_base) const noexcept {
    const auto it = entries_.find(entry);
    if (it == entries_.end()) { return false; }
    return restore_state_is_live(epoch, entry, it->second.committed_generation,
                                   it->second.claim_generation, reuse, reuse_base);
}

bool KVDiskCache::restore_state_is_live(std::uint64_t epoch, std::uint64_t entry,
                                          std::uint64_t committed_generation,
                                          std::uint64_t claim_generation, PrefixReusePath reuse,
                                          std::uint32_t reuse_base) const noexcept {
    if (!restore_active_ || !restore_target_.has_value() || restore_entry_id_ != entry ||
        restore_epoch_ != epoch || restore_target_->reuse != reuse ||
        restore_target_->reuse_base != reuse_base) {
        return false;
    }
    const auto it = entries_.find(entry);
    if (it == entries_.end()) { return false; }
    return it->second.committed_generation == committed_generation &&
           it->second.claim_generation == claim_generation;
}

void KVDiskCache::maybe_restore_state_barrier() {
    if (!restore_state_barrier_armed_.load(std::memory_order_acquire)) { return; }
    restore_state_entered_.store(true, std::memory_order_release);
    while (restore_state_barrier_armed_.load(std::memory_order_acquire) &&
           !restore_state_continue_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KVDiskCache::maybe_direct_state_read_barrier() {
    if (!direct_state_read_barrier_armed_.load(std::memory_order_acquire)) { return; }
    direct_state_read_entered_.store(true, std::memory_order_release);
    while (direct_state_read_barrier_armed_.load(std::memory_order_acquire) &&
           !direct_state_read_continue_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void KVDiskCache::finish_window_inflight(std::uint32_t slot, bool keep_slot) noexcept {
    std::lock_guard lock(mutex_);
    if (window_inflight_ > 0) { --window_inflight_; }
    if (!keep_slot && slot < window_slots()) { reset_window_slot_keep_host(window_[slot]); }
    idle_cv_.notify_all();
    cv_.notify_all();
}

void KVDiskCache::maybe_state_decode_stall() {
    state_decode_count_.fetch_add(1, std::memory_order_relaxed);
    const int ms = state_decode_stall_ms_.load(std::memory_order_acquire);
    if (ms > 0) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
}

void KVDiskCache::drop_spill(SpillSession& session, std::unique_lock<std::mutex>& lock) {
    purge_spill_jobs(session.epoch);
    if (session.failed || session.committed) { return; }
    if (!session.meta_installed) {
        for (std::size_t i = 0; i < session.new_object_ids.size(); ++i) {
            const std::uint64_t id = session.new_object_ids[i];
            objects_.erase(id);
        }
        for (std::uint64_t id : branch_shared_ids_) { drop_ref(id); }
        branch_shared_ids_.clear();
    }
    if (session.ticket != 0) { unpin_disk(session.ticket); }
    if (session.parent_id != 0 && session.parent_id != session.ticket) {
        unpin_disk(session.parent_id);
    }
    if (config_.ram != nullptr && session.ram_id != 0) {
        try {
            config_.ram->unpin_for_io(session.ram_id);
        } catch (...) {}
        auto it = ram_notes_.find(session.ram_id);
        if (it != ram_notes_.end()) {
            it->second.failed_this_generation = true;
            it->second.generation_stamp       = durable_generation_;
        }
    }
    session.failed = true;
    if (!session.cancelled) { ++drops_; }
    flush_queued_unlinks(lock);
}

bool KVDiskCache::idle_rewrite_of(std::uint64_t entry_id) const noexcept {
    return spill_ && !spill_->emergency && spill_->child_id == entry_id &&
           (spill_->action == SpillSession::Action::Extend ||
            spill_->action == SpillSession::Action::Refresh);
}

bool KVDiskCache::should_abandon_commit(const SpillSession& session) const noexcept {
    if (session.cancelled) { return true; }
    if (session.emergency) { return false; }
    if (session.action != SpillSession::Action::Extend &&
        session.action != SpillSession::Action::Refresh) {
        return false;
    }
    const auto it = entries_.find(session.child_id);
    return it != entries_.end() && it->second.pinned;
}

void KVDiskCache::write_meta_bin(std::uint64_t entry_id, const DiskMeta& meta, bool* renamed) {
    if (renamed != nullptr) { *renamed = false; }
    if (fail_rollback_meta_) {
        fail_rollback_meta_ = false;
        throw std::runtime_error("injected rollback meta write failure");
    }
    const auto meta_bytes = encode_meta(meta);
    if (meta_bytes.size() < 16 || std::memcmp(meta_bytes.data(), kDiskMetaMagic, 8) != 0) {
        throw std::runtime_error("KV disk meta encoding failed");
    }
    const auto dir = entry_dir(entry_id);
    std::filesystem::create_directories(dir);
    const auto tmp = config_.location / "tmp" / ("meta-" + std::to_string(entry_id));
    write_file_bytes(tmp, meta_bytes.data(), meta_bytes.size());
    fsync_path(tmp);
    std::filesystem::rename(tmp, dir / "meta.bin");
    if (renamed != nullptr) { *renamed = true; }
    if (fail_after_rollback_rename_) {
        fail_after_rollback_rename_ = false;
        throw std::runtime_error("injected post-rollback-rename failure");
    }
    fsync_dir(dir);
    fsync_dir(dir.parent_path());
}

void KVDiskCache::persist_checkpoints(SpillSession& session, std::unique_lock<std::mutex>& lock) {
    auto yield = [&]() -> bool {
        if (session.emergency) { return false; }
        if (session.cancelled) { return true; }
        return false;
    };
    if (yield()) { return; }
    auto pack_gdn = [&](const std::uint8_t* conv, const std::uint8_t* rec, DiskStateKind kind)
        -> std::uint64_t {
        if (yield()) { return 0; }
        const std::size_t conv_n = session.image.gdn_conv_bytes;
        const std::size_t rec_n  = session.image.gdn_recurrent_bytes;
        if (conv_n == 0 && rec_n == 0) { return 0; }
        if ((conv == nullptr && conv_n != 0) || (rec == nullptr && rec_n != 0)) { return 0; }
        if (session.next_encoded_state >= session.encoded_state.size() ||
            session.encoded_state[session.next_encoded_state].kind != kind) {
            throw std::logic_error("KV disk prepared state order mismatch");
        }
        const std::uint64_t id =
            write_state_blob(session.encoded_state[session.next_encoded_state++], lock);
        session.new_object_ids.push_back(id);
        session.new_object_kinds.push_back(DiskObjectKind::State);
        return id;
    };
    auto pack_buf = [&](const std::uint8_t* data, std::size_t n, DiskStateKind kind) -> std::uint64_t {
        if (yield()) { return 0; }
        if (data == nullptr || n == 0) { return 0; }
        if (session.next_encoded_state >= session.encoded_state.size() ||
            session.encoded_state[session.next_encoded_state].kind != kind) {
            throw std::logic_error("KV disk prepared state order mismatch");
        }
        const std::uint64_t id =
            write_state_blob(session.encoded_state[session.next_encoded_state++], lock);
        session.new_object_ids.push_back(id);
        session.new_object_kinds.push_back(DiskObjectKind::State);
        return id;
    };
    session.draft.current_gdn_id = pack_gdn(session.image.gdn_conv_current,
                                            session.image.gdn_recurrent_current, DiskStateKind::CurrentGdn);
    session.draft.current_hidden_id =
        pack_buf(session.image.tail_hidden, session.image.hidden_bytes, DiskStateKind::TailHidden);
    session.draft.current_cyclic_id =
        pack_buf(session.image.dflash_local, session.image.cyclic_bytes, DiskStateKind::DflashLocal);
    session.draft.rewrite_gdn_id     = 0;
    session.draft.rewrite_hidden_id  = 0;
    session.draft.rewrite_cyclic_id  = 0;
    if (session.host.rewrite_valid) {
        session.draft.rewrite_gdn_id =
            pack_gdn(session.image.gdn_conv_checkpoint, session.image.gdn_recurrent_checkpoint,
                     DiskStateKind::RewriteGdn);
        session.draft.rewrite_hidden_id = pack_buf(
            session.image.rewrite_hidden, session.image.rewrite_hidden_bytes, DiskStateKind::RewriteHidden);
        session.draft.rewrite_cyclic_id = pack_buf(
            session.image.dflash_rewrite, session.image.cyclic_bytes, DiskStateKind::DflashRewrite);
    }
    std::vector<RamLadderImage> ladders;
    std::vector<RamLadderImage> rollbacks;
    for (const RamLadderImage& image : session.image.ladder_images) {
        if (image.kind == ContextCheckpointKind::TurnRollback) {
            rollbacks.push_back(image);
        } else if (image.kind == ContextCheckpointKind::Ladder) {
            ladders.push_back(image);
        }
    }
    std::sort(ladders.begin(), ladders.end(),
              [](const RamLadderImage& a, const RamLadderImage& b) { return a.frontier > b.frontier; });
    std::sort(rollbacks.begin(), rollbacks.end(),
              [](const RamLadderImage& a, const RamLadderImage& b) { return a.frontier > b.frontier; });
    auto store_slot = [&](const RamLadderImage& image, DiskStateKind gdn_kind, DiskStateKind hidden_kind,
                          DiskStateKind cyclic_kind) {
        DiskCheckpointSlot slot;
        slot.frontier = image.frontier;
        slot.hash     = image.hash;
        slot.kind     = image.kind;
        slot.gdn_id   = pack_gdn(static_cast<const std::uint8_t*>(image.conv),
                                 static_cast<const std::uint8_t*>(image.recurrent), gdn_kind);
        slot.hidden_id =
            pack_buf(static_cast<const std::uint8_t*>(image.hidden), image.hidden_bytes, hidden_kind);
        slot.dflash_id =
            pack_buf(static_cast<const std::uint8_t*>(image.dflash), image.dflash_bytes, cyclic_kind);
        return slot;
    };
    session.draft.rollback = {};
    if (!rollbacks.empty()) {
        session.draft.rollback =
            store_slot(rollbacks.front(), DiskStateKind::RollbackGdn, DiskStateKind::RollbackHidden,
                       DiskStateKind::DflashRollback);
    }
    session.draft.ladders = {};
    for (std::size_t i = 0; i < ladders.size() && i < 2; ++i) {
        session.draft.ladders[i] =
            store_slot(ladders[i], DiskStateKind::LadderGdn, DiskStateKind::LadderHidden,
                       DiskStateKind::DflashLadder);
    }
    if (!session.cancelled && session.next_encoded_state != session.encoded_state.size()) {
        throw std::logic_error("KV disk prepared state was not fully consumed");
    }
}

void KVDiskCache::spill_one_page(SpillSession& session, std::uint32_t pool,
                                 std::uint32_t logical, std::unique_lock<std::mutex>& lock) {
    const Job job{.kind = session.emergency ? JobKind::EmergencySpillPage
                                            : JobKind::IdleSpillPage,
                  .ram_entry_id = session.ram_id,
                  .spill_epoch = session.epoch,
                  .pool = pool,
                  .logical_index = logical};
    spill_page_batch(session, std::span<const Job>(&job, 1), lock);
}

void KVDiskCache::spill_page_batch(SpillSession& session, std::span<const Job> jobs,
                                   std::unique_lock<std::mutex>& lock) {
    if (jobs.empty() || session.cancelled) { return; }
    const std::uint32_t pool = jobs.front().pool;
    const bool main = pool == 0;
    const PagedKVPool* kv_pool = main ? config_.text_pool : config_.backend_pool;
    if (kv_pool == nullptr) { return; }
    const std::uint8_t* image = main ? session.image.text : session.image.backend;
    const std::uint32_t pages = main ? session.image.text_pages : session.image.backend_pages;
    if (image == nullptr) { return; }

    struct PageRecord {
        std::uint64_t id = 0;
        std::uint64_t extent = 0;
        std::uint64_t stored = 0;
        std::uint64_t logical = 0;
        std::uint32_t record_crc = 0;
        std::array<std::uint8_t, kDiskPageHeaderBytes> header{};
        std::vector<std::pair<const void*, std::size_t>> slices;
        std::array<std::uint8_t, kDiskPageIoAlignment> padding{};
        std::vector<std::pair<const void*, std::uint64_t>> parts;
    };
    std::vector<PageRecord> records;
    records.reserve(jobs.size());
    std::uint64_t total_extent = 0;
    std::size_t total_iov = 0;
    const std::size_t payload = paged_kv_logical_page_bytes(*kv_pool);
    const std::uint64_t extent = aligned_page_file_bytes(payload);
    const std::size_t padding_bytes =
        static_cast<std::size_t>(extent) - kDiskPageHeaderBytes - payload;
    for (const Job& job : jobs) {
        if (job.kind != jobs.front().kind || job.ram_entry_id != session.ram_id ||
            job.spill_epoch != session.epoch || job.pool != pool || job.logical_index >= pages ||
            (!records.empty() && job.logical_index !=
                                      jobs.front().logical_index + records.size())) {
            throw std::logic_error("KV disk page batch is not contiguous");
        }
        records.emplace_back();
        PageRecord& record = records.back();
        record.extent = extent;
        record.stored = kDiskPageHeaderBytes + payload;
        record.logical = payload;
        std::memcpy(record.header.data(), kDiskPageMagic, 8);
        const std::uint32_t version = kDiskFormatVersion;
        std::memcpy(record.header.data() + 8, &version, 4);
        ninfer::logical_page_host_slices(image, *kv_pool, pages, job.logical_index,
                                         record.slices);
        std::uint32_t page_crc = ~0U;
        record.parts.emplace_back(record.header.data(), kDiskPageHeaderBytes);
        for (const auto& slice : record.slices) {
            record.parts.emplace_back(slice.first, static_cast<std::uint64_t>(slice.second));
            page_crc = crc32c_update(page_crc, std::span<const std::uint8_t>(
                                                   static_cast<const std::uint8_t*>(slice.first),
                                                   slice.second));
        }
        page_crc = ~page_crc;
        std::memcpy(record.header.data() + 12, &page_crc, 4);
        record.record_crc = crc32c_combine(
            crc32c(std::span<const std::uint8_t>(record.header.data(), record.header.size())),
            page_crc, payload);
        if (padding_bytes != 0) {
            record.parts.emplace_back(record.padding.data(), padding_bytes);
        }
        total_iov += record.parts.size();
        if (total_extent > std::numeric_limits<std::uint64_t>::max() - extent) {
            throw std::logic_error("KV disk page batch extent overflow");
        }
        total_extent += extent;
    }
    if (total_iov == 0 || total_iov > IOV_MAX || total_extent > SSIZE_MAX ||
        total_extent > kPackSegmentBytes) {
        throw std::logic_error("KV disk page batch exceeds pwritev limits");
    }

    const std::size_t index = kind_index(main ? DiskObjectKind::Main : DiskObjectKind::Backend);
    std::uint32_t segment = pack_active_segment_[index];
    std::uint64_t offset = pack_active_tail_[index];
    if (offset != 0 && total_extent > kPackSegmentBytes - offset) {
        if (segment == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("KV disk pack segment number overflow");
        }
        segment = ++pack_active_segment_[index];
        offset = 0;
        std::filesystem::create_directories(pack_path(main ? DiskObjectKind::Main
                                                           : DiskObjectKind::Backend,
                                                       segment).parent_path());
        const int made = ::open(pack_path(main ? DiskObjectKind::Main : DiskObjectKind::Backend,
                                          segment).c_str(), O_RDWR | O_CREAT | O_EXCL, 0644);
        if (made < 0) { throw std::runtime_error("failed to create KV disk page batch segment"); }
        if (::close(made) != 0) { throw std::runtime_error("failed to close KV page batch segment"); }
    }
    try {
        for (PageRecord& record : records) {
            record.id = allocate_object(main ? DiskObjectKind::Main : DiskObjectKind::Backend,
                                        record.extent);
        }
    } catch (...) {
        for (const PageRecord& record : records) {
            if (record.id != 0) { objects_.erase(record.id); }
        }
        throw;
    }
    // Reserve the complete range before dropping the cache lock.  Failed or
    // cancelled batches therefore leave unreachable bytes, never a reusable
    // range that a later append could overwrite.
    pack_active_segment_[index] = segment;
    pack_active_tail_[index] = offset + total_extent;
    if (!lock.owns_lock()) { lock.lock(); }
    lock.unlock();
    try {
        maybe_payload_io_stall();
        const auto descriptor = acquire_pack_descriptor(active_generation_,
                                                        main ? DiskObjectKind::Main
                                                             : DiskObjectKind::Backend,
                                                        segment, false);
        std::vector<iovec> iov;
        iov.reserve(total_iov);
        for (const PageRecord& record : records) {
            for (const auto& [data, bytes] : record.parts) {
                iov.push_back({const_cast<void*>(data), static_cast<std::size_t>(bytes)});
            }
        }
        std::size_t first = 0;
        std::uint64_t done = 0;
        while (done < total_extent) {
            const ssize_t n = pwritev_with_test_limit(
                descriptor->buffered, iov.data() + first,
                static_cast<int>(iov.size() - first), static_cast<off_t>(offset + done),
                take_test_partial_pwritev_bytes());
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) { throw std::runtime_error("KV disk page batch pwrite failed"); }
            done += static_cast<std::uint64_t>(n);
            ssize_t left = n;
            while (left > 0 && first < iov.size()) {
                auto& current = iov[first];
                if (static_cast<std::size_t>(left) < current.iov_len) {
                    current.iov_base = static_cast<char*>(current.iov_base) + left;
                    current.iov_len -= static_cast<std::size_t>(left);
                    break;
                }
                left -= static_cast<ssize_t>(current.iov_len);
                ++first;
            }
        }
        maybe_test_fault(DiskFaultPoint::AfterRecordWrite);
    } catch (...) {
        if (!lock.owns_lock()) { lock.lock(); }
        for (const PageRecord& record : records) { objects_.erase(record.id); }
        throw;
    }
    if (!lock.owns_lock()) { lock.lock(); }
    const DiskObjectKind kind = main ? DiskObjectKind::Main : DiskObjectKind::Backend;
    auto& ids = main ? session.draft.main_page_ids : session.draft.backend_page_ids;
    for (std::size_t i = 0; i < records.size(); ++i) {
        PageRecord& record = records[i];
        auto object = objects_.find(record.id);
        if (object == objects_.end()) { throw std::runtime_error("KV disk page batch object vanished"); }
        object->second.location.generation = active_generation_;
        object->second.location.segment = segment;
        object->second.location.offset = offset + static_cast<std::uint64_t>(i) * extent;
        object->second.location.extent_bytes = record.extent;
        object->second.location.stored_bytes = record.stored;
        object->second.location.logical_bytes = record.logical;
        object->second.location.record_crc32c = record.record_crc;
        session.new_object_ids.push_back(record.id);
        session.new_object_kinds.push_back(kind);
        const std::uint32_t logical = jobs[i].logical_index;
        if (logical >= ids.size()) { ids.resize(logical + 1); }
        ids[logical] = record.id;
    }
}

void KVDiskCache::purge_spill_jobs(std::uint64_t epoch) {
    if (epoch == 0) { return; }
    auto stale = [epoch](const Job& job) { return job.spill_epoch == epoch; };
    emergency_q_.erase(std::remove_if(emergency_q_.begin(), emergency_q_.end(), stale),
                        emergency_q_.end());
    idle_q_.erase(std::remove_if(idle_q_.begin(), idle_q_.end(), stale), idle_q_.end());
}

bool KVDiskCache::draft_ready(const SpillSession& session) const {
    if (session.draft.main_page_ids.size() < session.main_pages ||
        session.draft.backend_page_ids.size() < session.backend_pages) {
        return false;
    }
    for (std::uint32_t i = 0; i < session.main_pages; ++i) {
        if (session.draft.main_page_ids[i] == 0) { return false; }
    }
    for (std::uint32_t i = 0; i < session.backend_pages; ++i) {
        if (session.draft.backend_page_ids[i] == 0) { return false; }
    }
    if (session.host.execution_frontier != 0 &&
        (session.draft.ledger_id == 0 || session.draft.identity_id == 0)) {
        return false;
    }
    const bool need_gdn = config_.fingerprint.gdn_conv_bytes != 0 ||
                          config_.fingerprint.gdn_recurrent_bytes != 0;
    const bool need_cyclic = config_.fingerprint.cyclic_lane_bytes != 0;
    if (need_gdn && session.draft.current_gdn_id == 0) { return false; }
    if (need_cyclic && session.draft.current_cyclic_id == 0) { return false; }
    if (session.host.rewrite_valid) {
        if (need_gdn && session.draft.rewrite_gdn_id == 0) { return false; }
        if (need_cyclic && session.draft.rewrite_cyclic_id == 0) { return false; }
    }
    return true;
}

void KVDiskCache::record_uncertainty(std::uint64_t entry_id,
                                       const std::vector<std::uint64_t>& ids) {
    auto it = entries_.find(entry_id);
    if (it == entries_.end()) { return; }
    for (std::uint64_t id : ids) {
        if (id == 0) { continue; }
        add_hold(id);
        it->second.uncertainty_ids.push_back(id);
    }
}

void KVDiskCache::reclaim_orphan_objects() {
    std::error_code ec;
    std::unordered_set<std::uint64_t> indexed;
    indexed.reserve(entries_.size() + skipped_.size());
    for (const auto& [id, record] : entries_) {
        (void)record;
        indexed.insert(id);
    }
    for (const SkippedTree& tree : skipped_) { indexed.insert(tree.entry_id); }

    bool unknown_ownership = false;
    bool unindexed_extra   = false;
    for (const auto& entry :
         std::filesystem::directory_iterator(config_.location / "entries", ec)) {
        if (!entry.is_directory()) { continue; }
        std::uint64_t id = 0;
        try {
            id = static_cast<std::uint64_t>(std::stoull(entry.path().filename().string()));
        } catch (...) { continue; }
        if (id == 0 || entry_is_tombstoned(id) || indexed.count(id) != 0) { continue; }
        const auto meta_path = entry.path() / "meta.bin";
        if (!std::filesystem::exists(meta_path, ec)) { continue; }
        try {
            const auto decoded = try_decode_meta(read_file_bytes(meta_path));
            if (!decoded) {
                unknown_ownership = true;
            } else {
                unindexed_extra = true;
            }
        } catch (...) { unknown_ownership = true; }
        if (unknown_ownership || unindexed_extra) { break; }
    }
    if (unknown_ownership || unindexed_extra) { return; }

    std::uint32_t tombstone_files = 0;
    const std::uint32_t tombstone_cap = index_entry_limit();
    for (const auto& entry :
         std::filesystem::directory_iterator(config_.location / "tombstones", ec)) {
        if (!entry.is_regular_file()) { continue; }
        if (++tombstone_files > tombstone_cap) { return; }
    }

    std::unordered_set<std::uint64_t> referenced;
    auto add_id = [&](std::uint64_t id) {
        if (id != 0) { referenced.insert(id); }
    };
    for (const auto& [id, record] : entries_) {
        (void)id;
        std::vector<std::pair<DiskObjectKind, std::uint64_t>> objs;
        append_meta_objects(objs, record.meta);
        for (const auto& [kind, oid] : objs) {
            (void)kind;
            add_id(oid);
        }
        for (std::uint64_t held : record.uncertainty_ids) { add_id(held); }
    }
    for (const SkippedTree& tree : skipped_) {
        for (const auto& [kind, id] : tree.objects) {
            (void)kind;
            add_id(id);
        }
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(config_.location / "tombstones", ec)) {
        if (!entry.is_regular_file()) { continue; }
        try {
            const auto objects = try_decode_tombstone(read_file_bytes(entry.path()));
            if (!objects) { continue; }
            for (const auto& [kind, id] : *objects) {
                (void)kind;
                add_id(id);
            }
        } catch (...) { continue; }
    }
    for (const auto& [id, ref] : objects_) {
        if (ref.live_refs != 0 || ref.skip_refs != 0 || ref.hold_refs != 0) { add_id(id); }
    }
    static constexpr DiskObjectKind kKinds[] = {
        DiskObjectKind::Main, DiskObjectKind::Backend, DiskObjectKind::State,
        DiskObjectKind::Ledger, DiskObjectKind::Identity};
    for (DiskObjectKind kind : kKinds) {
        for (const auto& entry :
             std::filesystem::directory_iterator(config_.location / "objects" / kind_dir(kind),
                                                ec)) {
            if (!entry.is_regular_file()) { continue; }
            std::uint64_t id = 0;
            try {
                id = static_cast<std::uint64_t>(std::stoull(entry.path().filename().string()));
            } catch (...) { continue; }
            if (id == 0 || referenced.count(id) != 0) { continue; }
            queue_unlink(entry.path());
            objects_.erase(id);
        }
    }
}

void KVDiskCache::promote_idle_spill_to_emergency() {
    if (!spill_ || spill_->emergency || spill_->cancelled) { return; }
    spill_->emergency = true;
    for (auto it = idle_q_.begin(); it != idle_q_.end();) {
        if (it->spill_epoch != spill_->epoch) {
            ++it;
            continue;
        }
        if (it->kind == JobKind::IdleSpillPage) { it->kind = JobKind::EmergencySpillPage; }
        else if (it->kind == JobKind::IdleCommit) { it->kind = JobKind::EmergencyCommit; }
        emergency_q_.push_back(*it);
        it = idle_q_.erase(it);
    }
    idle_cv_.notify_all();
    cv_.notify_all();
}

void KVDiskCache::enqueue_spill_jobs(SpillSession& session) {
    const JobKind page_kind =
        session.emergency ? JobKind::EmergencySpillPage : JobKind::IdleSpillPage;
    const JobKind commit_kind =
        session.emergency ? JobKind::EmergencyCommit : JobKind::IdleCommit;
    for (std::uint32_t i = session.next_main; i < session.main_pages; ++i) {
        enqueue(Job{.kind = page_kind,
                    .ram_entry_id = session.ram_id,
                    .spill_epoch = session.epoch,
                    .pool = 0,
                    .logical_index = i});
    }
    for (std::uint32_t i = session.next_backend; i < session.backend_pages; ++i) {
        enqueue(Job{.kind = page_kind,
                    .ram_entry_id = session.ram_id,
                    .spill_epoch = session.epoch,
                    .pool = 1,
                    .logical_index = i});
    }
    enqueue(Job{.kind = commit_kind, .ram_entry_id = session.ram_id, .spill_epoch = session.epoch});
}

bool KVDiskCache::prepare_spill(std::uint64_t ram_id, bool emergency,
                                std::unique_lock<std::mutex>& lock) {
    if (config_.ram == nullptr) { return false; }
    if (!emergency && restore_cancels_idle_locked()) { return false; }
    auto note_it = ram_notes_.find(ram_id);
    if (note_it == ram_notes_.end()) { return false; }
    if (note_it->second.durable) { return false; }
    if (note_it->second.failed_this_generation &&
        note_it->second.generation_stamp == durable_generation_ && !emergency) {
        return false;
    }
    if (spill_) {
        if (spill_->ram_id == ram_id) {
            if (emergency) { promote_idle_spill_to_emergency(); }
            return true;
        }
        if (!spill_->emergency) {
            spill_->cancelled = true;
            drop_spill(*spill_, lock);
            spill_.reset();
        } else {
            return false;
        }
    }
    // Never make an emergency eviction fail merely because a queued restore
    // still owns a generation descriptor.  Maintenance is retried at the next
    // quiescent admission point; the hard logical capacity check below remains
    // authoritative for the new spill.
    // Compaction is scheduled only from explicit quiescent maintenance.  A
    // spill admission must stay independent of stale worker queue bookkeeping.
    SpillSession session;
    bool acquired_disk_pin = false;
    try {
    session.ram_id    = ram_id;
    session.ticket    = note_it->second.ticket;
    session.emergency = emergency;
    session.started   = std::chrono::steady_clock::now();
    session.image     = config_.ram->host_kv(ram_id);
    session.host      = config_.ram->load_host(ram_id);
    const std::uint64_t ticket = session.ticket;
    const bool ticket_live     = ticket != 0 && entries_.count(ticket) != 0;
    auto take_entry_id = [&]() -> std::optional<std::uint64_t> {
        if (next_entry_id_ == 0 || next_entry_id_ == std::numeric_limits<std::uint64_t>::max()) {
            return std::nullopt;
        }
        return next_entry_id_++;
    };
    if (!ticket_live) {
        const auto id = take_entry_id();
        if (!id) { return false; }
        session.action   = SpillSession::Action::Create;
        session.child_id = *id;
        session.draft.entry_id = session.child_id;
    } else {
        const IndexEntry& parent = require(ticket);
        const DiskMeta& committed = parent.meta;
        const std::uint32_t e     = committed.execution_frontier;
        const std::uint32_t e2    = session.host.execution_frontier;
        const PrefixHash128 captured_f =
            prefix_hash_at(session.host.ledger, session.host.identity, e2);
        const bool claimed_parent = parent.pinned;
        if (!claimed_parent && e2 == e && captured_f == committed.hash_f) {
            session.action   = SpillSession::Action::Refresh;
            session.child_id = ticket;
            session.draft    = committed;
        } else if (!claimed_parent && e2 > e &&
                   prefix_extension(session.host.ledger, parent.ledger, e)) {
            session.action    = SpillSession::Action::Extend;
            session.child_id  = ticket;
            session.parent_id = ticket;
            session.draft     = committed;
            session.share_tokens = e;
        } else {
            const auto id = take_entry_id();
            if (!id) { return false; }
            session.action       = SpillSession::Action::Branch;
            session.parent_id    = ticket;
            session.child_id     = *id;
            session.share_tokens = std::min(longest_ledger_prefix(session.host.ledger, parent.ledger),
                                              std::min(e, e2));
            session.draft = committed;
            session.draft.entry_id = session.child_id;
            session.draft.main_page_ids.clear();
            session.draft.backend_page_ids.clear();
        }
    }

    session.draft.execution_frontier      = session.host.execution_frontier;
    session.draft.ledger_frontier         = session.host.ledger_frontier;
    session.draft.rope_delta              = session.host.rope_delta;
    session.draft.text_kv_valid           = session.host.text_kv_valid;
    session.draft.mtp_kv_valid            = session.host.mtp_kv_valid;
    session.draft.dflash_context_frontier = session.host.dflash_context_frontier;
    session.draft.tail_hidden_valid       = session.host.tail_hidden_valid;
    session.draft.rewrite_valid           = session.host.rewrite_valid;
    session.draft.rewrite_kind            = session.host.rewrite_kind;
    session.draft.rewrite_frontier        = session.host.rewrite_frontier;
    session.draft.hash_c_valid            = session.host.rewrite_valid;
    session.draft.hash_f =
        prefix_hash_at(session.host.ledger, session.host.identity, session.host.execution_frontier);
    if (session.host.rewrite_valid && session.host.rewrite_frontier != 0) {
        session.draft.hash_c = prefix_hash_at(session.host.ledger, session.host.identity,
                                              session.host.rewrite_frontier);
    }

    const std::uint32_t share = session.action == SpillSession::Action::Create ||
                                        session.action == SpillSession::Action::Refresh
                                    ? 0
                                    : session.share_tokens;
    const std::uint32_t share_main =
        session.action == SpillSession::Action::Create ? 0
                                                       : share_pages(config_.fingerprint.speculative, true, share);
    const std::uint32_t share_backend =
        session.action == SpillSession::Action::Create
            ? 0
            : share_pages(config_.fingerprint.speculative, false, share);
    const auto aligned_share = [](std::uint32_t token_f, std::uint32_t n) {
        if (n == 0) { return 0U; }
        return (token_f % static_cast<std::uint32_t>(kPagedKVPageSize) == 0) ? n : n - 1U;
    };
    const std::uint32_t main_token_f = share;
    const std::uint32_t backend_token_f =
        config_.fingerprint.speculative == SpeculativeBackend::Mtp
            ? (share == 0 ? 0U : share - 1U)
            : share;
    const std::uint32_t keep_main     = aligned_share(main_token_f, share_main);
    const std::uint32_t keep_backend  = aligned_share(backend_token_f, share_backend);
    const auto cap_pages = [](std::uint32_t mapped, std::uint32_t tokens) {
        return std::min(mapped, ninfer::pages_for_tokens(tokens));
    };
    session.main_pages = cap_pages(session.image.text_pages,
                                   session.host.text_kv_valid != 0 ? session.host.text_kv_valid
                                                                   : session.host.execution_frontier);
    if (config_.backend_pool == nullptr) {
        session.backend_pages = 0;
    } else {
        switch (config_.fingerprint.speculative) {
        case SpeculativeBackend::Mtp:
            session.backend_pages =
                cap_pages(session.image.backend_pages, session.host.mtp_kv_valid);
            break;
        case SpeculativeBackend::DFlash:
            session.backend_pages =
                cap_pages(session.image.backend_pages,
                          session.host.dflash_context_frontier != 0
                              ? session.host.dflash_context_frontier
                              : session.host.execution_frontier);
            break;
        default:
            session.backend_pages = 0;
            break;
        }
    }
    if (session.action == SpillSession::Action::Refresh) {
        session.next_main    = session.main_pages;
        session.next_backend = session.backend_pages;
    } else {
        session.next_main    = keep_main;
        session.next_backend = keep_backend;
    }

    if (session.action == SpillSession::Action::Branch ||
        session.action == SpillSession::Action::Extend) {
        const IndexEntry& parent = require(session.parent_id != 0 ? session.parent_id : session.ticket);
        session.draft.main_page_ids.assign(parent.meta.main_page_ids.begin(),
                                           parent.meta.main_page_ids.begin() +
                                               std::min<std::size_t>(keep_main,
                                                                     parent.meta.main_page_ids.size()));
        session.draft.backend_page_ids.assign(
            parent.meta.backend_page_ids.begin(),
            parent.meta.backend_page_ids.begin() +
                std::min<std::size_t>(keep_backend, parent.meta.backend_page_ids.size()));
        if (session.action == SpillSession::Action::Branch) {
            for (std::uint32_t i = 0; i < keep_main && i < parent.meta.main_page_ids.size(); ++i) {
                add_ref(parent.meta.main_page_ids[i]);
                branch_shared_ids_.push_back(parent.meta.main_page_ids[i]);
            }
            for (std::uint32_t i = 0; i < keep_backend && i < parent.meta.backend_page_ids.size();
                 ++i) {
                add_ref(parent.meta.backend_page_ids[i]);
                branch_shared_ids_.push_back(parent.meta.backend_page_ids[i]);
            }
        }
        pin_disk(session.parent_id != 0 ? session.parent_id : session.ticket);
        session.ticket = session.parent_id != 0 ? session.parent_id : session.ticket;
        acquired_disk_pin = true;
    } else if (session.action == SpillSession::Action::Refresh) {
        pin_disk(session.child_id);
        session.ticket = session.child_id;
        acquired_disk_pin = true;
    }
    if (fail_prepare_spill_) {
        fail_prepare_spill_ = false;
        throw std::runtime_error("injected prepare_spill failure");
    }

    session.draft.main_page_ids.resize(session.main_pages);
    session.draft.backend_page_ids.resize(session.backend_pages);

    const std::uint32_t write_main =
        session.main_pages > session.next_main ? session.main_pages - session.next_main : 0;
    const std::uint32_t write_backend =
        session.backend_pages > session.next_backend ? session.backend_pages - session.next_backend
                                                      : 0;
    const std::uint64_t main_page_bytes =
        aligned_page_file_bytes(config_.text_pool != nullptr
                                    ? paged_kv_logical_page_bytes(*config_.text_pool)
                                    : config_.logical_page_bytes);
    const std::uint64_t backend_page_bytes =
        aligned_page_file_bytes(config_.backend_pool != nullptr
                                    ? paged_kv_logical_page_bytes(*config_.backend_pool)
                                    : config_.logical_page_bytes);
    const std::uint64_t page_bytes = static_cast<std::uint64_t>(write_main) * main_page_bytes +
                                     static_cast<std::uint64_t>(write_backend) * backend_page_bytes;
    std::uint64_t state_bytes = 0;
    auto prepare_state = [&](DiskStateKind kind, const void* first, std::size_t first_bytes,
                             const void* second = nullptr, std::size_t second_bytes = 0) {
        if (first_bytes == 0 && second_bytes == 0) { return; }
        session.encoded_state.push_back(
            encode_state_blob(kind, first, first_bytes, second, second_bytes));
        state_bytes += aligned_state_file_bytes(session.encoded_state.back().payload_bytes());
    };
    const std::size_t conv_n = session.image.gdn_conv_bytes;
    const std::size_t rec_n = session.image.gdn_recurrent_bytes;
    prepare_state(DiskStateKind::CurrentGdn, session.image.gdn_conv_current, conv_n,
                  session.image.gdn_recurrent_current, rec_n);
    prepare_state(DiskStateKind::TailHidden, session.image.tail_hidden,
                  session.image.hidden_bytes);
    prepare_state(DiskStateKind::DflashLocal, session.image.dflash_local,
                  session.image.cyclic_bytes);
    if (session.host.rewrite_valid) {
        prepare_state(DiskStateKind::RewriteGdn, session.image.gdn_conv_checkpoint, conv_n,
                      session.image.gdn_recurrent_checkpoint, rec_n);
        prepare_state(DiskStateKind::RewriteHidden, session.image.rewrite_hidden,
                      session.image.rewrite_hidden_bytes);
        prepare_state(DiskStateKind::DflashRewrite, session.image.dflash_rewrite,
                      session.image.cyclic_bytes);
    }
    std::vector<RamLadderImage> rollbacks;
    std::vector<RamLadderImage> ladders;
    for (const RamLadderImage& image : session.image.ladder_images) {
        if (image.kind == ContextCheckpointKind::TurnRollback) {
            rollbacks.push_back(image);
        } else if (image.kind == ContextCheckpointKind::Ladder) {
            ladders.push_back(image);
        }
    }
    std::sort(ladders.begin(), ladders.end(),
              [](const RamLadderImage& a, const RamLadderImage& b) { return a.frontier > b.frontier; });
    std::sort(rollbacks.begin(), rollbacks.end(),
              [](const RamLadderImage& a, const RamLadderImage& b) { return a.frontier > b.frontier; });
    auto prepare_slot = [&](const RamLadderImage& image, DiskStateKind gdn_kind,
                            DiskStateKind hidden_kind, DiskStateKind cyclic_kind) {
        prepare_state(gdn_kind, image.conv, conv_n, image.recurrent, rec_n);
        prepare_state(hidden_kind, image.hidden, image.hidden_bytes);
        prepare_state(cyclic_kind, image.dflash, image.dflash_bytes);
    };
    if (!rollbacks.empty()) {
        prepare_slot(rollbacks.front(), DiskStateKind::RollbackGdn,
                     DiskStateKind::RollbackHidden, DiskStateKind::DflashRollback);
    }
    for (std::size_t i = 0; i < ladders.size() && i < 2; ++i) {
        prepare_slot(ladders[i], DiskStateKind::LadderGdn, DiskStateKind::LadderHidden,
                     DiskStateKind::DflashLadder);
    }
    const auto aligned_raw_bytes = [](std::uint64_t bytes) -> std::uint64_t {
        if (bytes > std::numeric_limits<std::uint64_t>::max() -
                         (kDiskPageIoAlignment - 1)) {
            throw std::logic_error("KV disk raw object extent overflows");
        }
        return (bytes + (kDiskPageIoAlignment - 1)) &
               ~(static_cast<std::uint64_t>(kDiskPageIoAlignment) - 1);
    };
    if (session.host.ledger.size() >
        std::numeric_limits<std::uint64_t>::max() / sizeof(TokenId)) {
        throw std::logic_error("KV disk ledger size overflows");
    }
    const std::uint64_t ledger_bytes = aligned_raw_bytes(
        static_cast<std::uint64_t>(session.host.ledger.size()) * sizeof(TokenId));
    const std::uint64_t identity_bytes =
        aligned_raw_bytes(static_cast<std::uint64_t>(session.host.identity.packed_bytes()));
    if (page_bytes > std::numeric_limits<std::uint64_t>::max() - ledger_bytes ||
        page_bytes + ledger_bytes > std::numeric_limits<std::uint64_t>::max() - identity_bytes ||
        page_bytes + ledger_bytes + identity_bytes >
            std::numeric_limits<std::uint64_t>::max() - state_bytes) {
        throw std::logic_error("KV disk spill extent total overflows");
    }
    // Physical append bytes are immutable until a generation rewrite.  Keep
    // this full extent total separate from logical overwrite credit below.
    const std::uint64_t append_bytes = page_bytes + ledger_bytes + identity_bytes + state_bytes;
    std::uint64_t credit = 0;
    if (session.action == SpillSession::Action::Refresh ||
        session.action == SpillSession::Action::Extend) {
        const DiskMeta& replaced = require(session.child_id).meta;
        auto unique_bytes = [&](std::uint64_t id) -> std::uint64_t {
            if (id == 0) { return 0; }
            const auto it = objects_.find(id);
            if (it == objects_.end() || it->second.live_refs != 1) { return 0; }
            return it->second.bytes;
        };
        credit += unique_bytes(replaced.ledger_id);
        credit += unique_bytes(replaced.identity_id);
        credit += unique_bytes(replaced.current_gdn_id);
        credit += unique_bytes(replaced.current_hidden_id);
        credit += unique_bytes(replaced.current_cyclic_id);
        credit += unique_bytes(replaced.rewrite_gdn_id);
        credit += unique_bytes(replaced.rewrite_hidden_id);
        credit += unique_bytes(replaced.rewrite_cyclic_id);
        credit += unique_bytes(replaced.rollback.gdn_id);
        credit += unique_bytes(replaced.rollback.hidden_id);
        credit += unique_bytes(replaced.rollback.dflash_id);
        for (const auto& slot : replaced.ladders) {
            credit += unique_bytes(slot.gdn_id);
            credit += unique_bytes(slot.hidden_id);
            credit += unique_bytes(slot.dflash_id);
        }
        std::unordered_set<std::uint64_t> draft_pages;
        for (std::uint64_t id : session.draft.main_page_ids) {
            if (id != 0) { draft_pages.insert(id); }
        }
        for (std::uint64_t id : session.draft.backend_page_ids) {
            if (id != 0) { draft_pages.insert(id); }
        }
        for (std::uint64_t id : replaced.main_page_ids) {
            if (draft_pages.count(id) == 0) { credit += unique_bytes(id); }
        }
        for (std::uint64_t id : replaced.backend_page_ids) {
            if (draft_pages.count(id) == 0) { credit += unique_bytes(id); }
        }
    }
    const std::uint64_t logical_needed = append_bytes > credit ? append_bytes - credit : 0;
    // Garbage is physical until a generation rewrite completes.  Compact at
    // the admission boundary, before accepting any new append allocation,
    // rather than allowing the append path to grow past the 25% threshold.
    if (packs_need_compaction() && !compact_packs(lock, append_bytes)) {
        if (session.ticket != 0) { unpin_disk(session.ticket); }
        for (std::uint64_t id : branch_shared_ids_) { drop_ref(id); }
        branch_shared_ids_.clear();
        note_it->second.failed_this_generation = true;
        note_it->second.generation_stamp = durable_generation_;
        ++drops_;
        flush_queued_unlinks(lock);
        return false;
    }
    // Logical eviction cannot release append-pack blocks.  Refuse admission
    // before any record write if the filesystem cannot hold this append plus
    // the worst-case copy-on-write compaction generation and one segment of
    // rollover slack.
    constexpr std::uint64_t kPackSlack = 1ULL << 30;
    const std::uint64_t retained = packed_retained_bytes();
    const std::uint64_t reserve_base = packs_need_compaction() ? retained : 0;
    const auto available_bytes = [&]() -> std::optional<std::uint64_t> {
        if (test_free_bytes_override_) { return *test_free_bytes_override_; }
        struct statvfs fs {};
        if (::statvfs(config_.location.c_str(), &fs) != 0 || fs.f_frsize == 0 ||
            fs.f_bavail > std::numeric_limits<std::uint64_t>::max() / fs.f_frsize) {
            return std::nullopt;
        }
        return static_cast<std::uint64_t>(fs.f_bavail) * fs.f_frsize;
    }();
    const bool physical_room = available_bytes &&
                                reserve_base <= std::numeric_limits<std::uint64_t>::max() - append_bytes &&
                                reserve_base + append_bytes <=
                                    std::numeric_limits<std::uint64_t>::max() - kPackSlack &&
                                *available_bytes >= reserve_base + append_bytes + kPackSlack;
    if (!physical_room) {
        if (session.ticket != 0) { unpin_disk(session.ticket); }
        for (std::uint64_t id : branch_shared_ids_) { drop_ref(id); }
        branch_shared_ids_.clear();
        note_it->second.failed_this_generation = true;
        note_it->second.generation_stamp = durable_generation_;
        ++drops_;
        flush_queued_unlinks(lock);
        return false;
    }
    if (!make_capacity(logical_needed, lock)) {
        if (session.ticket != 0) { unpin_disk(session.ticket); }
        for (std::uint64_t id : branch_shared_ids_) { drop_ref(id); }
        branch_shared_ids_.clear();
        note_it->second.failed_this_generation = true;
        note_it->second.generation_stamp       = durable_generation_;
        ++drops_;
        flush_queued_unlinks(lock);
        return false;
    }

    session.epoch = next_spill_epoch_++;
    if (next_spill_epoch_ == 0) { next_spill_epoch_ = 1; }
    spill_ = std::move(session);
    enqueue_spill_jobs(*spill_);
    return true;
    } catch (...) {
        if (spill_ && spill_->ram_id == ram_id && !spill_->committed && !spill_->failed) {
            drop_spill(*spill_, lock);
            spill_.reset();
        } else {
            if (acquired_disk_pin) {
                if (session.ticket != 0) { unpin_disk(session.ticket); }
                if (session.parent_id != 0 && session.parent_id != session.ticket) {
                    unpin_disk(session.parent_id);
                }
            }
            for (std::uint64_t id : branch_shared_ids_) { drop_ref(id); }
            branch_shared_ids_.clear();
            flush_queued_unlinks(lock);
        }
        auto failed = ram_notes_.find(ram_id);
        if (failed != ram_notes_.end()) {
            failed->second.failed_this_generation = true;
            failed->second.generation_stamp       = durable_generation_;
        }
        ++drops_;
        idle_cv_.notify_all();
        cv_.notify_all();
        return false;
    }
}

void KVDiskCache::commit_spill(SpillSession& session, std::unique_lock<std::mutex>& lock) {
    if (session.cancelled || session.failed) {
        drop_spill(session, lock);
        return;
    }
    if (crash_before_meta_) {
        crash_before_meta_ = false;
        drop_spill(session, lock);
        return;
    }
    persist_checkpoints(session, lock);
    if (should_abandon_commit(session)) {
        drop_spill(session, lock);
        return;
    }
    {
        std::vector<std::uint8_t> ledger(session.host.ledger.size() * sizeof(TokenId));
        if (!session.host.ledger.empty()) {
            std::memcpy(ledger.data(), session.host.ledger.data(), ledger.size());
        }
        session.draft.ledger_id = allocate_object(DiskObjectKind::Ledger, ledger.size());
        write_new_object(DiskObjectKind::Ledger, session.draft.ledger_id, ledger.data(),
                         ledger.size(), &lock);
        session.new_object_ids.push_back(session.draft.ledger_id);
        session.new_object_kinds.push_back(DiskObjectKind::Ledger);
    }
    if (should_abandon_commit(session)) {
        drop_spill(session, lock);
        return;
    }
    {
        std::vector<std::uint8_t> identity(session.host.identity.packed_bytes());
        session.host.identity.pack(identity.data());
        session.draft.identity_id = allocate_object(DiskObjectKind::Identity, identity.size());
        write_new_object(DiskObjectKind::Identity, session.draft.identity_id, identity.data(),
                         identity.size(), &lock);
        session.new_object_ids.push_back(session.draft.identity_id);
        session.new_object_kinds.push_back(DiskObjectKind::Identity);
    }
    if (should_abandon_commit(session)) {
        drop_spill(session, lock);
        return;
    }
    if (!draft_ready(session)) {
        drop_spill(session, lock);
        return;
    }
    session.draft.entry_id = session.child_id;
    const auto meta_bytes  = encode_meta(session.draft);
    if (meta_bytes.size() < 16 || std::memcmp(meta_bytes.data(), kDiskMetaMagic, 8) != 0) {
        drop_spill(session, lock);
        return;
    }
    const auto dir = entry_dir(session.child_id);
    std::filesystem::create_directories(dir);
    const auto tmp = config_.location / "tmp" / ("meta-" + std::to_string(session.child_id));
    bool renamed = false;
    objects_dir_fsynced_.store(false, std::memory_order_release);
    lock.unlock();
    try {
        fsync_new_objects(session);
        objects_dir_fsynced_.store(true, std::memory_order_release);
        write_file_bytes(tmp, meta_bytes.data(), meta_bytes.size());
        maybe_test_fault(DiskFaultPoint::AfterMetaTmpWrite);
        fsync_path(tmp);
        maybe_test_fault(DiskFaultPoint::AfterMetaTmpSync);
        std::filesystem::rename(tmp, dir / "meta.bin");
        renamed = true;
        meta_renamed_.store(true, std::memory_order_release);
        maybe_test_fault(DiskFaultPoint::AfterMetaRename);
        if (stall_after_meta_rename_) {
            stall_after_meta_rename_ = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (fail_after_meta_rename_) {
            fail_after_meta_rename_ = false;
            throw std::runtime_error("injected post-rename failure");
        }
        fsync_dir(dir);
        maybe_test_fault(DiskFaultPoint::AfterEntryDirSync);
        fsync_dir(dir.parent_path());
        maybe_test_fault(DiskFaultPoint::AfterEntryParentSync);
        fsync_dir(config_.location / "tmp");
        fsync_dir(config_.location);
    } catch (...) {
        if (!lock.owns_lock()) { lock.lock(); }
        session.meta_installed = renamed;
        if (!renamed) { throw; }
        install_committed_entry(session, lock, false, true);
        return;
    }
    if (!lock.owns_lock()) { lock.lock(); }
        session.meta_installed = true;
    if (should_abandon_commit(session)) {
        const bool had_entry = entries_.count(session.child_id) != 0;
        if (had_entry) {
            const DiskMeta previous = entries_[session.child_id].meta;
            bool rolled_back = false;
            lock.unlock();
            try {
                write_meta_bin(session.child_id, previous, &rolled_back);
            } catch (...) {
                if (!lock.owns_lock()) { lock.lock(); }
                if (!rolled_back) {
                    install_committed_entry(session, lock);
                    return;
                }
                session.meta_installed = true;
                session.committed       = true;
                record_uncertainty(session.child_id, session.new_object_ids);
                release_spill_pins(session, true);
                purge_spill_jobs(session.epoch);
                return;
            }
            lock.lock();
            session.meta_installed = false;
        } else {
            session.meta_installed = false;
            queue_unlink(entry_dir(session.child_id));
        }
        drop_spill(session, lock);
        if (!had_entry) {
            if (lock.owns_lock()) { lock.unlock(); }
            try {
                fsync_store_dirs();
            } catch (...) { ++drops_; }
            if (!lock.owns_lock()) { lock.lock(); }
        }
        return;
    }
    install_committed_entry(session, lock);
}

void KVDiskCache::release_spill_pins(SpillSession& session, bool mark_failed) {
    if (session.ticket != 0) { unpin_disk(session.ticket); }
    if (session.parent_id != 0 && session.parent_id != session.ticket) {
        unpin_disk(session.parent_id);
    }
    if (config_.ram != nullptr && session.ram_id != 0) {
        try {
            config_.ram->unpin_for_io(session.ram_id);
        } catch (...) {}
        auto it = ram_notes_.find(session.ram_id);
        if (it != ram_notes_.end() && mark_failed) {
            it->second.failed_this_generation = true;
            it->second.generation_stamp       = durable_generation_;
            it->second.durable               = false;
        }
    }
    idle_cv_.notify_all();
}

void KVDiskCache::install_committed_entry(SpillSession& session, std::unique_lock<std::mutex>& lock,
                                            bool mark_durable, bool retain_replaced) {
    DiskMeta replaced;
    std::vector<std::uint64_t> prev_uncertainty;
    const bool had = entries_.count(session.child_id) != 0;
    bool pinned          = false;
    std::uint32_t io_pins = 0;
    std::uint64_t claim_generation = 0;
    std::uint64_t committed_generation = 0;
    if (had) {
        replaced = entries_[session.child_id].meta;
        prev_uncertainty = entries_[session.child_id].uncertainty_ids;
        pinned   = entries_[session.child_id].pinned;
        io_pins  = entries_[session.child_id].io_pins;
        claim_generation = entries_[session.child_id].claim_generation;
        committed_generation = entries_[session.child_id].committed_generation;
    }
    IndexEntry record;
    record.meta     = session.draft;
    record.ledger   = session.host.ledger;
    record.identity = session.host.identity;
    record.pinned   = pinned;
    record.io_pins  = io_pins;
    record.claim_generation = claim_generation;
    record.committed_generation = had ? std::max(committed_generation, std::uint64_t{1}) + 1 : 1;
    if (retain_replaced) { record.uncertainty_ids = prev_uncertainty; }
    entries_[session.child_id] = std::move(record);
    idle_cv_.wait(lock, [&] {
        if (stopping_) { return true; }
        for (const WindowSlot& slot : window_) {
            if (slot.disk_entry_id == session.child_id && slot.assigned && !slot.filled) {
                return false;
            }
        }
        return true;
    });
    for (WindowSlot& slot : window_) {
        if (slot.disk_entry_id == session.child_id) { reset_window_slot_keep_host(slot); }
    }
    if (!had) { fifo_.push_back(session.child_id); }
    auto publish = [&](DiskObjectKind kind, std::uint64_t id) {
        if (id == 0) { return; }
        objects_[id].kind = kind;
        add_ref(id);
    };
    for (std::size_t i = 0; i < session.new_object_ids.size(); ++i) {
        publish(session.new_object_kinds[i], session.new_object_ids[i]);
    }
    if ((session.action == SpillSession::Action::Extend ||
         session.action == SpillSession::Action::Refresh) &&
        had) {
        std::unordered_set<std::uint64_t> kept;
        std::vector<std::pair<DiskObjectKind, std::uint64_t>> draft_objs;
        append_meta_objects(draft_objs, session.draft);
        for (const auto& [kind, id] : draft_objs) {
            (void)kind;
            kept.insert(id);
        }
        std::vector<std::pair<DiskObjectKind, std::uint64_t>> old_objs;
        append_meta_objects(old_objs, replaced);
        for (const auto& [kind, id] : old_objs) {
            (void)kind;
            if (id == 0 || kept.count(id) != 0) { continue; }
            if (retain_replaced) {
                release_live_ref(id);
                add_hold(id);
                entries_[session.child_id].uncertainty_ids.push_back(id);
            } else {
                drop_ref(id);
            }
        }
    }
    if (!retain_replaced) {
        for (std::uint64_t id : prev_uncertainty) { release_hold(id); }
    }
    branch_shared_ids_.clear();
    session.committed = true;
    bump_version();
    bump_durable_generation();
    if (session.ticket != 0) { unpin_disk(session.ticket); }
    if (config_.ram != nullptr) {
        try {
            config_.ram->set_disk_entry_id(session.ram_id, session.child_id);
        } catch (...) {
            try {
                config_.ram->set_disk_entry_id(session.ram_id, session.child_id);
            } catch (...) {}
        }
        try {
            config_.ram->unpin_for_io(session.ram_id);
        } catch (...) {}
    }
    auto note = ram_notes_.find(session.ram_id);
    if (note != ram_notes_.end()) {
        note->second.ticket  = session.child_id;
        note->second.durable = mark_durable;
        note->second.failed_this_generation = !mark_durable;
        note->second.generation_stamp       = durable_generation_;
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                       session.started)
                             .count();
    save_seconds_ += elapsed;
    pending_save_seconds_ += elapsed;
    ++captures_;
    try {
        write_manifest(lock);
    } catch (...) {}
    flush_queued_unlinks(lock);
}

bool KVDiskCache::emergency_spill_ram(std::uint64_t ram_id) {
    if (config_.ram == nullptr) { return false; }
    auto wait_done = [&](std::unique_lock<std::mutex>& lock, bool allow_terminal = false) {
        idle_cv_.wait(lock, [&] {
            if (!spill_ || spill_->ram_id != ram_id) { return true; }
            return allow_terminal && stopping_ &&
                   (spill_->committed || spill_->failed || spill_->cancelled);
        });
        return ram_notes_.count(ram_id) != 0 && ram_notes_[ram_id].durable;
    };
    auto unpin_extra = [&] {
        try {
            config_.ram->unpin_for_io(ram_id);
        } catch (...) {}
    };
    std::unique_lock lock(mutex_);
    idle_cancel_rams_.insert(ram_id);
    idle_cv_.wait(lock, [&] {
        return stopping_ ||
               (!(idle_pinning_ && idle_pinning_ram_ == ram_id) &&
                !(spill_ && spill_->ram_id == ram_id &&
                  (spill_->cancelled || spill_->failed || spill_->committed)));
    });
    if (stopping_) {
        idle_cancel_rams_.erase(ram_id);
        if (spill_ && spill_->ram_id == ram_id) { return wait_done(lock, true); }
        return false;
    }
    if (spill_ && spill_->ram_id == ram_id) {
        promote_idle_spill_to_emergency();
        idle_cancel_rams_.erase(ram_id);
        return wait_done(lock);
    }
    lock.unlock();
    try {
        config_.ram->pin_for_io(ram_id);
    } catch (...) {
        std::lock_guard inner(mutex_);
        idle_cancel_rams_.erase(ram_id);
        return false;
    }
    try {
        while (!config_.ram->copies_ready(ram_id)) { config_.ram->wait_pending_copies(); }
    } catch (...) {
        unpin_extra();
        std::lock_guard inner(mutex_);
        idle_cancel_rams_.erase(ram_id);
        return false;
    }
    lock.lock();
    auto note = ram_notes_.find(ram_id);
    if (note != ram_notes_.end() && note->second.durable) {
        idle_cancel_rams_.erase(ram_id);
        lock.unlock();
        unpin_extra();
        return true;
    }
    idle_cv_.wait(lock, [&] {
        return payload_io_inflight_ == 0 || stopping_ || !spill_ || spill_->ram_id == ram_id;
    });
    if (spill_ && spill_->ram_id == ram_id && !spill_->cancelled && !spill_->failed &&
        !spill_->committed) {
        promote_idle_spill_to_emergency();
        idle_cancel_rams_.erase(ram_id);
        lock.unlock();
        unpin_extra();
        lock.lock();
        return wait_done(lock);
    }
    bool prepared = false;
    try {
        prepared = prepare_spill(ram_id, true, lock);
    } catch (...) {
        prepared = false;
        auto it = ram_notes_.find(ram_id);
        if (it != ram_notes_.end()) {
            it->second.failed_this_generation = true;
            it->second.generation_stamp       = durable_generation_;
        }
        ++drops_;
    }
    idle_cancel_rams_.erase(ram_id);
    if (!prepared) {
        lock.unlock();
        unpin_extra();
        return false;
    }
    return wait_done(lock);
}

void KVDiskCache::flush_not_durable_ram(
    const std::function<void(std::uint64_t done, std::uint64_t total)>& on_progress) {
    if (config_.ram == nullptr) { return; }
    std::uint64_t total = 0;
    {
        std::lock_guard lock(mutex_);
        for (const auto& [id, note] : ram_notes_) {
            if (!note.durable) { ++total; }
        }
    }
    if (on_progress) { on_progress(0, total); }
    std::unordered_set<std::uint64_t> attempted;
    std::uint64_t done = 0;
    for (;;) {
        std::uint64_t ram_id = 0;
        {
            std::lock_guard lock(mutex_);
            for (const auto& [id, note] : ram_notes_) {
                if (!note.durable && attempted.count(id) == 0) {
                    ram_id = id;
                    break;
                }
            }
        }
        if (ram_id == 0) { break; }
        attempted.insert(ram_id);
        (void)emergency_spill_ram(ram_id);
        std::lock_guard lock(mutex_);
        auto it = ram_notes_.find(ram_id);
        if (it != ram_notes_.end() && !it->second.durable) {
            it->second.failed_this_generation = true;
            it->second.generation_stamp       = durable_generation_;
        }
        ++done;
        if (on_progress) { on_progress(done, total); }
    }
}

void KVDiskCache::wait_idle_and_fsync() {
    std::unique_lock lock(mutex_);
    idle_requested_ = false;
    idle_cv_.wait(lock, [&] {
        return !spill_ && restore_q_.empty() && emergency_q_.empty() && idle_q_.empty() &&
               prefetch_q_.empty() && !idle_pinning_ && window_inflight_ == 0 &&
               payload_io_inflight_ == 0 && restore_state_inflight_ == 0 &&
               reader_claims_.empty();
    });
    if (packs_need_compaction()) { (void)compact_packs(lock); }
    reap_retired_generations();
    lock.unlock();
    try {
        write_manifest();
        if (packset_publication_pending_sync_) {
            maybe_test_fault(DiskFaultPoint::CompactionRecoveryBeforeRootSync);
        }
        fsync_dir(config_.location);
        lock.lock();
        packset_publication_pending_sync_ = false;
        reap_retired_generations();
        lock.unlock();
    } catch (...) { ++drops_; }
}

void KVDiskCache::discard_prefetch_queue() {
    for (const Job& job : prefetch_q_) { unpin_disk(job.disk_entry_id); }
    prefetch_q_.clear();
}

void KVDiskCache::purge_prefetch_of(std::uint64_t entry_id) {
    prefetch_q_.erase(std::remove_if(prefetch_q_.begin(), prefetch_q_.end(),
                                     [&](const Job& job) {
                                         if (job.disk_entry_id != entry_id) { return false; }
                                         unpin_disk(job.disk_entry_id);
                                         return true;
                                     }),
                      prefetch_q_.end());
}

void KVDiskCache::publish_page_job_failure(const Job& job) {
    ++drops_;
    if (page_job_is_live_restore(job)) { restore_failed_ = true; }
}

void KVDiskCache::reset_window_slot_keep_host(WindowSlot& slot) noexcept {
    void* io             = slot.io;
    void* host           = slot.host;
    cudaEvent_t h2d_arrived = slot.h2d_arrived;
    cudaEvent_t h2d_event = slot.h2d_event;
    slot                  = {};
    slot.io               = io;
    slot.host             = host;
    slot.h2d_arrived      = h2d_arrived;
    slot.h2d_event        = h2d_event;
}

void KVDiskCache::close_restore_session_locked() {
    ++restore_epoch_;
    discard_prefetch_queue();
    restore_q_.clear();
    for (WindowSlot& slot : window_) { reset_window_slot_keep_host(slot); }
    restore_target_.reset();
    restore_entry_id_ = 0;
    restore_active_   = false;
    retire_copies_event_locked();
    idle_cv_.notify_all();
    cv_.notify_all();
}

void KVDiskCache::try_close_restore_session_locked() {
    reclaim_completed_h2d_slots();
    if (!restore_kv_done_ || !restore_state_done_ || restore_failed_ || window_inflight_ != 0) {
        return;
    }
    for (const WindowSlot& slot : window_) {
        if (slot.assigned || slot.filled) { return; }
    }
    restore_active_ = false;
    close_restore_session_locked();
}

void KVDiskCache::fill_window_slot(std::uint32_t slot, DiskObjectKind kind, std::uint64_t object_id,
                                   std::uint32_t pool, std::uint32_t logical) {
    const std::size_t payload =
        pool == 0 ? (config_.text_pool ? paged_kv_logical_page_bytes(*config_.text_pool) : 0)
                  : (config_.backend_pool ? paged_kv_logical_page_bytes(*config_.backend_pool) : 0);
    void* io = nullptr;
    void* host = nullptr;
    std::uint64_t entry = 0;
    std::uint64_t epoch = 0;
    std::uint64_t claim = 0;
    std::uint64_t committed = 0;
    {
        std::lock_guard lock(mutex_);
        WindowSlot& win = window_[slot];
        if (!win.assigned || win.pool != pool || win.logical_index != logical ||
            win.disk_entry_id == 0) {
            return;
        }
        io        = win.io;
        host      = win.host;
        entry     = win.disk_entry_id;
        epoch     = win.epoch;
        claim     = win.claim_generation;
        committed = win.committed_generation;
    }
    const auto io_started = std::chrono::steady_clock::now();
    const bool ok = read_page_payload(kind, object_id, io, host, payload);
    const auto io_ended = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    WindowSlot& win = window_[slot];
    const bool same_job = win.assigned && win.disk_entry_id == entry && win.epoch == epoch &&
                           win.claim_generation == claim && win.committed_generation == committed &&
                           win.pool == pool && win.logical_index == logical;
    if (!same_job) {
        return;
    }
    const bool identity_live =
        !stopping_ && slot_matches_index(win) &&
        (!restore_target_ ||
         (win.epoch == restore_epoch_ && win.disk_entry_id == restore_entry_id_));
    if (ok && identity_live) { note_live_host_load_locked(io_started, io_ended); }
    if (!identity_live) {
        if (!ok) { ++drops_; }
        reset_window_slot_keep_host(win);
        idle_cv_.notify_all();
        return;
    }
    if (!ok) {
        ++drops_;
        if (slot_is_live_restore_page(win)) { restore_failed_ = true; }
        reset_window_slot_keep_host(win);
        idle_cv_.notify_all();
        return;
    }
    const auto it = entries_.find(win.disk_entry_id);
    if (it != entries_.end()) {
        const auto& ids =
            pool == 0 ? it->second.meta.main_page_ids : it->second.meta.backend_page_ids;
        if (logical >= ids.size() || ids[logical] != object_id) {
            reset_window_slot_keep_host(win);
            idle_cv_.notify_all();
            return;
        }
    }
    win.object_id = object_id;
    win.filled    = true;
    win.h2d_done  = false;
    idle_cv_.notify_all();
}

void KVDiskCache::prefetch_window(std::uint64_t entry_id, std::uint32_t text_dst_pages,
                                  std::uint32_t backend_dst_pages) {
    std::lock_guard lock(mutex_);
    const IndexEntry& record = require(entry_id);
    if (!record.pinned) { return; }
    const std::uint64_t claim_generation = record.claim_generation;
    std::uint32_t queued     = 0;
    auto push = [&](std::uint32_t pool, std::uint32_t logical, std::uint64_t object_id) {
        if (queued >= 2u || queued >= window_slots() || object_id == 0) { return; }
        pin_disk(entry_id);
        enqueue(Job{.kind                  = JobKind::PrefetchWindow,
                    .disk_entry_id        = entry_id,
                    .restore_epoch        = restore_epoch_,
                    .claim_generation      = claim_generation,
                    .committed_generation = record.committed_generation,
                    .pool                 = pool,
                    .logical_index        = logical,
                    .window_slot          = queued});
        ++queued;
    };
    const std::uint32_t text_n =
        std::min(text_dst_pages, static_cast<std::uint32_t>(record.meta.main_page_ids.size()));
    const std::uint32_t backend_n =
        std::min(backend_dst_pages, static_cast<std::uint32_t>(record.meta.backend_page_ids.size()));
    for (std::uint32_t i = 0; i < text_n && queued < 2u && queued < window_slots(); ++i) {
        push(0, i, record.meta.main_page_ids[i]);
    }
    for (std::uint32_t i = 0; i < backend_n && queued < 2u && queued < window_slots(); ++i) {
        push(1, i, record.meta.backend_page_ids[i]);
    }
}

std::uint64_t KVDiskCache::restore_device(std::uint64_t entry_id, const DiskRestoreTarget& target) {
    std::unique_lock lock(mutex_);
    idle_cv_.wait(lock, [&] {
        if (stopping_) { return true; }
        const auto it = entries_.find(entry_id);
        if (it == entries_.end()) { return true; }
        const std::uint64_t claim     = it->second.claim_generation;
        const std::uint64_t committed = it->second.committed_generation;
        for (const WindowSlot& slot : window_) {
            const bool keep = slot.assigned && slot.disk_entry_id == entry_id &&
                              slot.epoch == restore_epoch_ && slot.claim_generation == claim &&
                              slot.committed_generation == committed;
            if (slot.assigned && !slot.filled && !keep) { return false; }
        }
        return true;
    });
    if (stopping_) { return 0; }
    wait_state_arena_idle(lock);
    if (stopping_) { return 0; }
    prefetch_q_.erase(std::remove_if(prefetch_q_.begin(), prefetch_q_.end(),
                                     [&](const Job& job) {
                                         if (job.disk_entry_id == entry_id) { return false; }
                                         unpin_disk(job.disk_entry_id);
                                         return true;
                                     }),
                      prefetch_q_.end());
    restore_q_.erase(std::remove_if(restore_q_.begin(), restore_q_.end(),
                                    [&](const Job& job) { return job.disk_entry_id != entry_id; }),
                     restore_q_.end());
    const IndexEntry& record = require(entry_id);
    for (WindowSlot& slot : window_) {
        const bool keep = slot.assigned && slot.disk_entry_id == entry_id &&
                            slot.epoch == restore_epoch_ &&
                            slot.claim_generation == record.claim_generation &&
                            slot.committed_generation == record.committed_generation;
        if (!keep) {
            reset_window_slot_keep_host(slot);
            continue;
        }
        if (slot.h2d_done && slot.h2d_event != nullptr) {
            CUDA_CHECK(cudaEventSynchronize(slot.h2d_event));
        }
        slot.h2d_done = false;
    }
    restore_target_        = target;
    restore_entry_id_      = entry_id;
    restore_failed_        = false;
    restore_kv_done_       = false;
    restore_state_done_    = false;
    restore_state_loaded_  = false;
    restore_state_slices_  = {};
    restore_active_        = true;
    wait_ticket_epoch_     = restore_epoch_;
    copies_record_epoch_  = wait_ticket_epoch_;
    copies_ticket_refs_    = 1;
    restore_next_main_     = 0;
    restore_next_backend_  = 0;
    restore_h2d_main_      = 0;
    restore_h2d_backend_   = 0;
    copies_timed_          = false;
    copies_join_recorded_  = false;
    h2d_billed_            = false;
    if (target.text_dst_pages > record.meta.main_page_ids.size() ||
        target.backend_dst_pages > record.meta.backend_page_ids.size()) {
        restore_failed_ = true;
        ++drops_;
        restore_kv_done_    = true;
        restore_state_done_ = true;
        idle_cv_.notify_all();
        return wait_ticket_epoch_;
    }
    const std::uint32_t text_n    = target.text_dst_pages;
    const std::uint32_t backend_n  = target.backend_dst_pages;
    const std::uint64_t claim = record.claim_generation;
    const std::uint64_t committed = record.committed_generation;
    auto already = [&](std::uint32_t pool, std::uint32_t logical) {
        for (const WindowSlot& slot : window_) {
            if (slot.assigned && slot.disk_entry_id == entry_id && slot.pool == pool &&
                slot.logical_index == logical && slot.epoch == restore_epoch_ &&
                slot.claim_generation == claim && slot.committed_generation == committed) {
                return true;
            }
        }
        for (const Job& job : restore_q_) {
            if (job.disk_entry_id == entry_id && job.pool == pool && job.logical_index == logical &&
                job.restore_epoch == restore_epoch_ && job.claim_generation == claim &&
                job.committed_generation == committed) {
                return true;
            }
        }
        for (const Job& job : prefetch_q_) {
            if (job.disk_entry_id == entry_id && job.pool == pool && job.logical_index == logical &&
                job.restore_epoch == restore_epoch_ && job.claim_generation == claim &&
                job.committed_generation == committed) {
                return true;
            }
        }
        for (const ReaderClaim& claim_page : reader_claims_) {
            if (claim_page.disk_entry_id == entry_id && claim_page.restore_epoch == restore_epoch_ &&
                claim_page.claim_generation == claim &&
                claim_page.committed_generation == committed && claim_page.pool == pool &&
                claim_page.logical_index == logical) {
                return true;
            }
        }
        return false;
    };
    for (std::uint32_t i = 0; i < text_n; ++i) {
        if (already(0, i)) { continue; }
        enqueue(Job{.kind                  = JobKind::RestoreRead,
                    .disk_entry_id        = entry_id,
                    .restore_epoch        = restore_epoch_,
                    .claim_generation      = claim,
                    .committed_generation = committed,
                    .pool                 = 0,
                    .logical_index        = i},
                &lock);
    }
    for (std::uint32_t i = 0; i < backend_n; ++i) {
        if (already(1, i)) { continue; }
        enqueue(Job{.kind                  = JobKind::RestoreRead,
                    .disk_entry_id        = entry_id,
                    .restore_epoch        = restore_epoch_,
                    .claim_generation      = claim,
                    .committed_generation = committed,
                    .pool                 = 1,
                    .logical_index        = i},
                &lock);
    }
    restore_use_context_head_ = is_staged_checkpoint_restore(target.reuse) && target.reuse_base != 0;
    restore_unpack_rewrite_ =
        !restore_use_context_head_ || record.meta.rewrite_frontier <= target.reuse_base;
    restore_checkpoint_host_ = {};
    if (text_n == 0 && backend_n == 0) { restore_kv_done_ = true; }
    if (restore_q_.empty()) { cv_.notify_all(); }
    return wait_ticket_epoch_;
}

void KVDiskCache::reclaim_completed_h2d_slots() {
    for (WindowSlot& slot : window_) {
        if (!slot.assigned || !slot.h2d_done) { continue; }
        if (slot.h2d_event == nullptr) {
            reset_window_slot_keep_host(slot);
            continue;
        }
        const cudaError_t ready = cudaEventQuery(slot.h2d_event);
        if (ready == cudaErrorNotReady) { continue; }
        CUDA_CHECK(ready);
        reset_window_slot_keep_host(slot);
    }
}

void KVDiskCache::wait_h2d_slots_locked() {
    for (WindowSlot& slot : window_) {
        if (!slot.h2d_done || slot.h2d_event == nullptr) { continue; }
        CUDA_CHECK(cudaEventSynchronize(slot.h2d_event));
    }
}

void KVDiskCache::h2d_ready_slots(cudaStream_t stream) {
    if (!restore_target_) { return; }
    reclaim_completed_h2d_slots();
    DiskRestoreTarget& target = *restore_target_;
    for (std::size_t slot_index = 0; slot_index < window_.size(); ++slot_index) {
        WindowSlot& slot = window_[slot_index];
        if (!slot.filled || slot.h2d_done) { continue; }
        const std::uint32_t extent =
            slot.pool == 0 ? target.text_dst_pages : target.backend_dst_pages;
        if (slot.logical_index >= extent || !slot_is_live_restore_page(slot)) {
            reset_window_slot_keep_host(slot);
            continue;
        }
        if (!copies_timed_) {
            ensure_copies_start_locked();
            if (copies_start_ != nullptr) {
                CUDA_CHECK(cudaEventRecord(copies_start_, stream));
                copies_timed_ = true;
            }
        }
        auto* staging = static_cast<unsigned char*>(page_staging_) +
                        slot_index * page_staging_stride_;
        PagedKVAllocation* allocation = nullptr;
        const PagedKVPool* pool = nullptr;
        PagedKVScatterPlane* planes = nullptr;
        std::size_t plane_count = 0;
        std::size_t max_plane_bytes = 0;
        std::size_t logical_bytes = 0;
        if (slot.pool == 0) {
            allocation     = target.text;
            pool           = target.text_pool;
            planes         = text_scatter_planes_;
            plane_count    = text_scatter_plane_count_;
            max_plane_bytes = text_scatter_max_plane_bytes_;
            logical_bytes  = text_scatter_page_bytes_;
        } else {
            allocation     = target.backend;
            pool           = target.backend_pool;
            planes         = backend_scatter_planes_;
            plane_count    = backend_scatter_plane_count_;
            max_plane_bytes = backend_scatter_max_plane_bytes_;
            logical_bytes  = backend_scatter_page_bytes_;
        }
        if (allocation == nullptr || pool == nullptr || planes == nullptr || logical_bytes == 0 ||
            logical_bytes > page_staging_stride_ || slot.h2d_arrived == nullptr ||
            slot.h2d_event == nullptr || page_scatter_stream_ == nullptr) {
            throw std::logic_error("KV disk page scatter resources are incomplete");
        }
        CUDA_CHECK(cudaMemcpyAsync(staging, slot.host, logical_bytes, cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaEventRecord(slot.h2d_arrived, stream));
        CUDA_CHECK(cudaStreamWaitEvent(page_scatter_stream_, slot.h2d_arrived, 0));
        scatter_paged_kv_logical_page_from_device(*allocation, *pool, staging, planes, plane_count,
                                                  max_plane_bytes, slot.logical_index,
                                                  page_scatter_stream_);
        maybe_scatter_record_barrier(slot.logical_index);
        CUDA_CHECK(cudaEventRecord(slot.h2d_event, page_scatter_stream_));
        if (slot.pool == 0) {
            restore_next_main_ = std::max(restore_next_main_, slot.logical_index + 1);
            ++restore_h2d_main_;
        } else {
            restore_next_backend_ = std::max(restore_next_backend_, slot.logical_index + 1);
            ++restore_h2d_backend_;
        }
        slot.h2d_done = true;
    }
    idle_cv_.notify_all();
    cv_.notify_all();
}

void KVDiskCache::wait_state_arena_idle(std::unique_lock<std::mutex>& lock) {
    if (!state_arena_h2d_pending_) { return; }
    if (state_h2d_stream_ == nullptr) {
        throw std::logic_error("KV disk immediate-state stream is missing");
    }
    lock.unlock();
    CUDA_CHECK(cudaStreamSynchronize(state_h2d_stream_));
    lock.lock();
    state_arena_h2d_pending_ = false;
}

void KVDiskCache::start_restore_state_h2d_locked(cudaStream_t stream) {
    if (!restore_target_ || restore_failed_) { return; }
    auto live = [&] {
        return restore_target_ && !restore_failed_ && !restore_state_done_;
    };
    if (!copies_timed_) {
        ensure_copies_start_locked();
        if (copies_start_ != nullptr) {
            CUDA_CHECK(cudaEventRecord(copies_start_, stream));
            copies_timed_ = true;
        }
    }
    bool copied_state = false;
    auto gdn_from_arena = [&](std::int32_t slot, const auto& conv, const auto& rec) {
        if (!live() || restore_target_->gdn == nullptr) { return; }
        if (conv.empty() && rec.empty()) { return; }
        state_arena_h2d_pending_ = true;
        restore_target_->gdn->unpack_slot_from_host(
            slot, conv.empty() ? nullptr : conv.data(), rec.empty() ? nullptr : rec.data(),
            state_h2d_stream_);
        copied_state = true;
    };
    auto hidden_from_arena = [&](Tensor* dst, const auto& src) {
        if (!live() || dst == nullptr || src.empty()) { return; }
        state_arena_h2d_pending_ = true;
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src.data(), src.size(), cudaMemcpyHostToDevice,
                                   state_h2d_stream_));
        copied_state = true;
    };
    auto cyclic_from_arena = [&](CyclicKVCache* cache, const auto& src) {
        if (!live() || cache == nullptr || src.empty()) { return; }
        state_arena_h2d_pending_ = true;
        cache->copy_lane_from_host(src.data(), restore_target_->dflash_lane,
                                   state_h2d_stream_);
        copied_state = true;
    };

    const bool skip_frontier_current = restore_target_ &&
        is_rewrite_checkpoint_restore(restore_target_->reuse) && !restore_use_context_head_;
    if (restore_use_context_head_) {
        gdn_from_arena(restore_target_->gdn_current_slot, restore_state_slices_.gdn_conv,
                       restore_state_slices_.gdn_rec);
        hidden_from_arena(restore_target_->tail_hidden, restore_state_slices_.hidden);
        cyclic_from_arena(restore_target_->dflash_local, restore_state_slices_.cyclic);
    } else if (!skip_frontier_current) {
        gdn_from_arena(restore_target_->gdn_current_slot, restore_state_slices_.gdn_conv,
                       restore_state_slices_.gdn_rec);
    }
    if (restore_unpack_rewrite_) {
        gdn_from_arena(restore_target_->gdn_checkpoint_slot,
                       restore_state_slices_.rewrite_gdn_conv,
                       restore_state_slices_.rewrite_gdn_rec);
    }
    if (!restore_use_context_head_ && !skip_frontier_current) {
        hidden_from_arena(restore_target_->tail_hidden, restore_state_slices_.hidden);
    }
    if (restore_unpack_rewrite_) {
        hidden_from_arena(restore_target_->rewrite_checkpoint_hidden,
                          restore_state_slices_.rewrite_hidden);
    }
    if (!restore_use_context_head_ && !skip_frontier_current) {
        cyclic_from_arena(restore_target_->dflash_local, restore_state_slices_.cyclic);
    }
    if (restore_unpack_rewrite_) {
        cyclic_from_arena(restore_target_->dflash_checkpoint,
                          restore_state_slices_.rewrite_cyclic);
    }
    if (fail_after_state_h2d_enqueue_.exchange(false, std::memory_order_acq_rel)) {
        throw std::runtime_error("injected failure after KV disk state H2D enqueue");
    }
    if (!live()) { return; }
    if (copied_state) {
        if (state_arena_idle_ == nullptr) {
            throw std::logic_error("KV disk immediate-state arena event is missing");
        }
        CUDA_CHECK(cudaEventRecord(state_arena_idle_, state_h2d_stream_));
        CUDA_CHECK(cudaStreamWaitEvent(stream, state_arena_idle_, 0));
    }
}

void KVDiskCache::finish_restore_state_locked(std::unique_lock<std::mutex>& lock,
                                             cudaStream_t stream) {
    if (!restore_target_ || restore_failed_) { return; }
    if (restore_state_done_) {
        if (restore_kv_done_) {
            record_restore_join_locked(stream);
            try_close_restore_session_locked();
        }
        return;
    }
    if (!restore_state_loaded_) { return; }
    if (restore_state_inflight_ != 0) { return; }
    ++restore_state_inflight_;
    bool own_inflight = true;
    auto release_h2d = [&] {
        if (!lock.owns_lock()) { lock.lock(); }
        if (own_inflight && restore_state_inflight_ > 0) {
            --restore_state_inflight_;
            own_inflight = false;
        }
        idle_cv_.notify_all();
        cv_.notify_all();
    };
    try {
        start_restore_state_h2d_locked(stream);
        restore_state_done_ = true;
        if (restore_kv_done_) {
            record_restore_join_locked(stream);
            try_close_restore_session_locked();
        }
        else {
            idle_cv_.notify_all();
        }
        release_h2d();
    } catch (...) {
        if (lock.owns_lock() && restore_target_ && !restore_failed_) {
            restore_failed_ = true;
            ++drops_;
        }
        release_h2d();
        throw;
    }
}

void KVDiskCache::finish_restore_state() {
    std::unique_lock lock(mutex_);
    if (restore_target_) { finish_restore_state_locked(lock, restore_target_->stream); }
}

void KVDiskCache::pump_restore_locked(std::unique_lock<std::mutex>& lock, cudaStream_t stream) {
    if (restore_failed_) { return; }
    reclaim_completed_h2d_slots();
    const cudaStream_t live = restore_target_ ? restore_target_->stream : stream;
    h2d_ready_slots(live);
    if (!restore_target_) { return; }
    const DiskRestoreTarget& target = *restore_target_;
    const IndexEntry& record        = require(restore_entry_id_);
    if (target.text_dst_pages > record.meta.main_page_ids.size() ||
        target.backend_dst_pages > record.meta.backend_page_ids.size()) {
        restore_failed_ = true;
        ++drops_;
        return;
    }
    const std::uint32_t text_n    = target.text_dst_pages;
    const std::uint32_t backend_n  = target.backend_dst_pages;
    if (restore_h2d_main_ >= text_n && restore_h2d_backend_ >= backend_n) {
        restore_kv_done_ = true;
    }
    finish_restore_state_locked(lock, live);
}

void KVDiskCache::pump_restore(cudaStream_t stream) {
    std::unique_lock lock(mutex_);
    pump_restore_locked(lock, stream);
}

void KVDiskCache::cancel_restore() {
    std::unique_lock lock(mutex_);
    ++restore_epoch_;
    restore_q_.clear();
    discard_prefetch_queue();
    restore_active_       = false;
    restore_target_.reset();
    restore_entry_id_     = 0;
    restore_failed_       = false;
    restore_kv_done_      = true;
    restore_state_done_   = true;
    restore_state_loaded_ = false;
    copies_join_recorded_ = false;
    restore_checkpoint_host_ = {};
    retire_copies_event_locked();
    idle_cv_.notify_all();
    cv_.notify_all();
    restore_state_continue_.store(true, std::memory_order_release);
    idle_cv_.wait(lock, [&] {
        return (window_inflight_ == 0 && restore_state_inflight_ == 0) || stopping_;
    });
    wait_state_arena_idle(lock);
    restore_state_slices_ = {};
    wait_h2d_slots_locked();
    for (WindowSlot& slot : window_) { reset_window_slot_keep_host(slot); }
}

bool KVDiskCache::copies_ready() const {
    std::lock_guard lock(mutex_);
    if (restore_failed_) { return false; }
    if (!restore_active_ && !restore_target_) { return true; }
    return false;
}

bool KVDiskCache::restore_failed() const {
    std::lock_guard lock(mutex_);
    return restore_failed_;
}

DiskRestoredHost KVDiskCache::take_restore_checkpoints() {
    std::lock_guard lock(mutex_);
    return std::move(restore_checkpoint_host_);
}

void KVDiskCache::wait_copies() { wait_copies(0); }

void KVDiskCache::wait_copies(std::uint64_t epoch) {
    if (fail_next_wait_copies_.exchange(false, std::memory_order_acq_rel)) {
        throw std::logic_error("injected wait_copies failure");
    }
    cudaEvent_t done = wait_restore_copy_event(epoch);
    maybe_copy_lease_barrier();
    if (done == nullptr) { return; }
    try {
        CUDA_CHECK(cudaEventSynchronize(done));
        bill_post_disk_h2d();
    } catch (...) {
        release_copy_event(done);
        throw;
    }
    release_copy_event(done);
}

void KVDiskCache::wait_copies_on_stream(cudaStream_t stream) { wait_copies_on_stream(stream, 0); }

void KVDiskCache::wait_copies_on_stream(cudaStream_t stream, std::uint64_t epoch) {
    cudaEvent_t done = wait_restore_copy_event(epoch);
    maybe_copy_lease_barrier();
    if (done == nullptr) { return; }
    try {
        CUDA_CHECK(cudaEventSynchronize(done));
        bill_post_disk_h2d();
        CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
    } catch (...) {
        release_copy_event(done);
        throw;
    }
    release_copy_event(done);
}

cudaEvent_t KVDiskCache::wait_restore_copy_event(std::uint64_t epoch) {
    std::uint64_t wait_epoch = epoch;
    bool saw_restore          = epoch != 0;
    bool latched_barrier       = false;
    for (;;) {
        std::unique_lock lock(mutex_);
        if (epoch != 0) {
            if (restore_failed_ && restore_epoch_ == epoch) {
                throw std::runtime_error("KV disk restore pread failed");
            }
            if (restore_epoch_ != epoch || (!restore_active_ && !restore_target_)) { break; }
            if (restore_kv_done_ && restore_state_loaded_ && restore_target_ &&
                restore_epoch_ == epoch) {
                finish_restore_state_locked(lock, restore_target_->stream);
            }
            if (!restore_target_ && restore_kv_done_ && restore_state_done_) { break; }
            if (restore_target_ && restore_epoch_ == epoch) {
                pump_restore_locked(lock, restore_target_->stream);
            }
            if (!latched_barrier) {
                latched_barrier = true;
                lock.unlock();
                maybe_wait_epoch_barrier();
                continue;
            }
            idle_cv_.wait_for(lock, std::chrono::milliseconds(2));
            continue;
        }
        if (saw_restore && restore_epoch_ != wait_epoch) { break; }
        if (restore_failed_ && (!saw_restore || restore_epoch_ == wait_epoch)) {
            throw std::runtime_error("KV disk restore pread failed");
        }
        if (!restore_active_ && !restore_target_) { break; }
        if (restore_kv_done_ && restore_state_loaded_ && restore_target_ &&
            (!saw_restore || restore_epoch_ == wait_epoch)) {
            finish_restore_state_locked(lock, restore_target_->stream);
        }
        if (!restore_target_ && restore_kv_done_ && restore_state_done_ &&
            (!saw_restore || restore_epoch_ == wait_epoch)) {
            break;
        }
        if (restore_target_ && (!saw_restore || restore_epoch_ == wait_epoch)) {
            wait_epoch  = restore_epoch_;
            saw_restore = true;
            pump_restore_locked(lock, restore_target_->stream);
        }
        if (saw_restore && !latched_barrier) {
            latched_barrier = true;
            lock.unlock();
            maybe_wait_epoch_barrier();
            continue;
        }
        idle_cv_.wait_for(lock, std::chrono::milliseconds(2));
    }
    std::lock_guard lock(mutex_);
    const std::uint64_t lease_epoch =
        epoch != 0 ? epoch : (saw_restore ? wait_epoch : wait_ticket_epoch_);
    last_wait_copy_event_ = lease_epoch != 0 ? lease_copy_event_locked(lease_epoch) : nullptr;
    return last_wait_copy_event_;
}

void KVDiskCache::record_copies_done_locked(cudaStream_t stream) {
    ensure_live_copies_event_locked();
    if (copies_done_ == nullptr) { return; }
    CUDA_CHECK(cudaEventRecord(copies_done_, stream));
    copies_record_epoch_ = restore_epoch_;
}

void KVDiskCache::record_restore_join_locked(cudaStream_t stream) {
    if (copies_join_recorded_ || !restore_kv_done_ || !restore_state_done_) { return; }
    for (const WindowSlot& slot : window_) {
        if (!slot.assigned || !slot.h2d_done || slot.h2d_event == nullptr) { continue; }
        CUDA_CHECK(cudaStreamWaitEvent(stream, slot.h2d_event, 0));
    }
    record_copies_done_locked(stream);
    copies_join_recorded_ = true;
}

void KVDiskCache::ensure_live_copies_event_locked() {
    if (copies_done_waiters_ > 0 && copies_done_ != nullptr) { retire_copies_event_locked(); }
    if (copies_done_ == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&copies_done_, cudaEventDefault));
        if (wait_ticket_epoch_ != 0 && wait_ticket_epoch_ == restore_epoch_) {
            copies_ticket_refs_ = 1;
        }
    }
}

void KVDiskCache::ensure_copies_start_locked() {
    if (copies_start_ == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&copies_start_, cudaEventDefault));
    }
}

void KVDiskCache::try_harvest_copy_pair_locked(cudaEvent_t start, cudaEvent_t done, bool& timed) {
    if (!timed || start == nullptr || done == nullptr) { return; }
    if (cudaEventQuery(done) != cudaSuccess) { return; }
    float ms = 0;
    if (cudaEventElapsedTime(&ms, start, done) != cudaSuccess) { return; }
    timed = false;
    ++timing_harvests_;
}

void KVDiskCache::bill_post_disk_h2d() {
    std::lock_guard lock(mutex_);
    if (h2d_billed_ || !load_host_end_) { return; }
    const auto now = std::chrono::steady_clock::now();
    if (now > *load_host_end_) {
        pending_h2d_seconds_ += std::chrono::duration<double>(now - *load_host_end_).count();
    }
    h2d_billed_ = true;
}

void KVDiskCache::note_live_host_load_locked(std::chrono::steady_clock::time_point started,
                                               std::chrono::steady_clock::time_point ended) {
    if (ended < started) { return; }
    if (!load_host_start_ || started < *load_host_start_) { load_host_start_ = started; }
    if (!load_host_end_ || ended > *load_host_end_) { load_host_end_ = ended; }
}

void KVDiskCache::harvest_retired_timing_locked() {
    for (RetiredCopyEvent& item : retired_copy_events_) {
        if (!item.timed) { continue; }
        try_harvest_copy_pair_locked(item.start, item.event, item.timed);
        if (!item.timed && item.start != nullptr) {
            (void)cudaEventDestroy(item.start);
            item.start = nullptr;
        }
    }
}

void KVDiskCache::retire_copies_event_locked() {
    if (copies_done_ == nullptr && copies_ticket_refs_ == 0) { return; }
    RetiredCopyEvent item;
    item.event   = copies_done_;
    item.epoch   = copies_record_epoch_ != 0 ? copies_record_epoch_ : wait_ticket_epoch_;
    item.waiters = copies_done_waiters_;
    item.ticket_refs = copies_ticket_refs_;
    if (copies_done_ != nullptr && copies_timed_) {
        bool timed = true;
        try_harvest_copy_pair_locked(copies_start_, copies_done_, timed);
        if (timed) {
            item.start = copies_start_;
            item.timed = true;
            copies_start_ = nullptr;
        }
        copies_timed_ = false;
    }
    retired_copy_events_.push_back(item);
    copies_done_         = nullptr;
    copies_done_waiters_ = 0;
    copies_ticket_refs_ = 0;
}

cudaEvent_t KVDiskCache::lease_copy_event_locked(std::uint64_t epoch) {
    if (copies_done_ != nullptr && copies_record_epoch_ == epoch) {
        ++copies_done_waiters_;
        return copies_done_;
    }
    for (auto it = retired_copy_events_.rbegin(); it != retired_copy_events_.rend(); ++it) {
        if (it->event != nullptr && it->epoch == epoch) {
            ++it->waiters;
            return it->event;
        }
    }
    return nullptr;
}

void KVDiskCache::release_copy_event(cudaEvent_t event) {
    if (event == nullptr) { return; }
    std::lock_guard lock(mutex_);
    if (event == copies_done_ && copies_done_waiters_ > 0) { --copies_done_waiters_; }
    harvest_retired_timing_locked();
    for (auto it = retired_copy_events_.begin(); it != retired_copy_events_.end();) {
        if (it->event != event) {
            ++it;
            continue;
        }
        if (it->waiters > 0) { --it->waiters; }
        if (it->waiters == 0 && it->ticket_refs == 0) {
            try_harvest_copy_pair_locked(it->start, it->event, it->timed);
            if (it->start != nullptr) { (void)cudaEventDestroy(it->start); }
            (void)cudaEventDestroy(it->event);
            it = retired_copy_events_.erase(it);
        } else {
            ++it;
        }
    }
}

void KVDiskCache::release_restore_ticket(std::uint64_t epoch) {
    if (epoch == 0) { return; }
    std::lock_guard lock(mutex_);
    if (copies_record_epoch_ == epoch && copies_ticket_refs_ > 0) { --copies_ticket_refs_; }
    harvest_retired_timing_locked();
    for (auto it = retired_copy_events_.begin(); it != retired_copy_events_.end();) {
        if (it->epoch == epoch && it->ticket_refs > 0) { --it->ticket_refs; }
        if (it->waiters == 0 && it->ticket_refs == 0) {
            if (it->event != nullptr) {
                try_harvest_copy_pair_locked(it->start, it->event, it->timed);
                if (it->start != nullptr) { (void)cudaEventDestroy(it->start); }
                (void)cudaEventDestroy(it->event);
            }
            it = retired_copy_events_.erase(it);
        } else {
            ++it;
        }
    }
}

void KVDiskCache::maybe_copy_lease_barrier() {
    if (!copy_lease_barrier_armed_.load(std::memory_order_acquire)) { return; }
    copy_event_leased_.store(true, std::memory_order_release);
    while (copy_lease_barrier_armed_.load(std::memory_order_acquire) &&
           !copy_lease_continue_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KVDiskCache::maybe_wait_epoch_barrier() {
    if (!wait_epoch_barrier_armed_.load(std::memory_order_acquire)) { return; }
    wait_epoch_latched_.store(true, std::memory_order_release);
    while (wait_epoch_barrier_armed_.load(std::memory_order_acquire) &&
           !wait_epoch_continue_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KVDiskCache::maybe_page_read_barrier() {
    page_read_count_.fetch_add(1, std::memory_order_relaxed);
    if (!page_read_barrier_armed_.load(std::memory_order_acquire)) { return; }
    page_read_entered_.store(true, std::memory_order_release);
    while (page_read_barrier_armed_.load(std::memory_order_acquire) &&
           !page_read_continue_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KVDiskCache::maybe_payload_take_barrier() {
    if (!payload_take_barrier_armed_.load(std::memory_order_acquire)) { return; }
    payload_take_entered_.store(true, std::memory_order_release);
    while (payload_take_barrier_armed_.load(std::memory_order_acquire) &&
           !payload_take_continue_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KVDiskCache::maybe_slot_assign_barrier(std::uint32_t logical_index) {
    if (logical_index == 0) { return; }
    if (!slot_assign_barrier_armed_.load(std::memory_order_acquire)) { return; }
    slot_assign_entered_.store(true, std::memory_order_release);
    while (slot_assign_barrier_armed_.load(std::memory_order_acquire) &&
           !slot_assign_continue_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KVDiskCache::maybe_scatter_record_barrier(std::uint32_t logical_index) {
    if (!scatter_record_barrier_armed_.load(std::memory_order_acquire) ||
        logical_index != scatter_record_logical_.load(std::memory_order_acquire)) {
        return;
    }
    scatter_record_entered_.store(true, std::memory_order_release);
    if (scatter_test_gate_ == nullptr || page_scatter_stream_ == nullptr) {
        throw std::logic_error("KV disk scatter test gate is incomplete");
    }
    CUDA_CHECK(cudaStreamWaitEvent(page_scatter_stream_, scatter_test_gate_, 0));
}

KvDiskCopySeconds KVDiskCache::harvest_copy_seconds() {
    std::lock_guard lock(mutex_);
    harvest_retired_timing_locked();
    try_harvest_copy_pair_locked(copies_start_, copies_done_, copies_timed_);
    KvDiskCopySeconds out;
    out.save              = pending_save_seconds_;
    pending_save_seconds_ = 0;
    out.h2d               = pending_h2d_seconds_;
    pending_h2d_seconds_  = 0;
    if (load_host_start_ && load_host_end_ && *load_host_end_ > *load_host_start_) {
        out.load = std::chrono::duration<double>(*load_host_end_ - *load_host_start_).count();
    }
    load_seconds_ += out.load;
    load_host_start_.reset();
    load_host_end_.reset();
    return out;
}

KvDiskSnapshot KVDiskCache::snapshot() const noexcept {
    std::lock_guard lock(mutex_);
    return KvDiskSnapshot{
        .capacity_bytes = config_.capacity_bytes,
        .used_bytes     = static_cast<std::size_t>(unique_bytes_),
        .entry_count    = entries_.size(),
        .captures       = captures_,
        .restores       = restores_,
        .evictions      = evictions_,
        .drops          = drops_,
        .save_seconds   = save_seconds_,
        .load_seconds   = load_seconds_,
    };
}

void KVDiskCache::load_restore_state_locked(std::unique_lock<std::mutex>& lock) {
    if (!restore_target_ || restore_state_loaded_ || restore_failed_ || restore_entry_id_ == 0) {
        return;
    }
    if (restore_state_inflight_ != 0) { return; }
    ++restore_state_inflight_;
    bool own_inflight = true;
    const std::uint64_t entry = restore_entry_id_;
    const std::uint64_t epoch = restore_epoch_;
    const PrefixReusePath reuse = restore_target_->reuse;
    const std::uint32_t reuse_base = restore_target_->reuse_base;
    std::uint64_t committed = 0;
    std::uint64_t claim      = 0;
    auto release_owner = [&] {
        if (!lock.owns_lock()) { lock.lock(); }
        if (own_inflight && restore_state_inflight_ > 0) {
            --restore_state_inflight_;
            own_inflight = false;
        }
        idle_cv_.notify_all();
        cv_.notify_all();
    };
    {
        const auto it = entries_.find(entry);
        if (it == entries_.end()) {
            restore_failed_ = true;
            ++drops_;
            release_owner();
            return;
        }
        committed = it->second.committed_generation;
        claim      = it->second.claim_generation;
    }
    auto publish_setup_failure = [&] {
        release_owner();
        if (restore_state_is_live(epoch, entry, committed, claim, reuse, reuse_base)) {
            restore_failed_ = true;
            ++drops_;
        }
    };
    DiskMeta meta;
    bool use_head       = false;
    bool unpack_rewrite = false;
    std::size_t conv_n  = 0;
    std::size_t rec_n   = 0;
    std::size_t hidden_n = 0;
    std::size_t rewrite_hidden_n = 0;
    std::size_t cyclic_n = 0;
    DiskCheckpointSlot slot{};
    ImmediateStateSlices state_slices;
    DiskRestoredHost checkpoint_host;
    CheckpointDecodePlan checkpoint_plan;
    std::jthread checkpoint_prepare;
    std::atomic<bool> checkpoint_prepare_finished{false};
    bool checkpoint_prepare_started = false;
    bool checkpoint_prepare_ok      = false;
    bool failed = false;
    bool stale  = false;
    std::optional<std::chrono::steady_clock::time_point> restore_io_started;
    auto bill_restore_io = [&] {
        if (!restore_io_started) { return; }
        const auto started = *restore_io_started;
        const auto ended   = std::chrono::steady_clock::now();
        restore_io_started.reset();
        if (ended < started) { return; }
        auto add_if_live = [&] {
            if (!restore_state_is_live(epoch, entry, committed, claim, reuse, reuse_base)) {
                return;
            }
            note_live_host_load_locked(started, ended);
        };
        if (lock.owns_lock()) {
            add_if_live();
        } else {
            std::lock_guard inner(mutex_);
            add_if_live();
        }
    };
    try {
    if (fail_next_restore_state_setup_.exchange(false, std::memory_order_acq_rel)) {
        throw std::runtime_error("injected KV disk restore state setup failure");
    }
    meta            = require(entry).meta;
    use_head        = restore_use_context_head_;
    unpack_rewrite  = restore_unpack_rewrite_;
    conv_n          = static_cast<std::size_t>(config_.fingerprint.gdn_conv_bytes);
    rec_n           = static_cast<std::size_t>(config_.fingerprint.gdn_recurrent_bytes);
    hidden_n =
        restore_target_->tail_hidden ? restore_target_->tail_hidden->bytes() : 0;
    rewrite_hidden_n = restore_target_->rewrite_checkpoint_hidden
                           ? restore_target_->rewrite_checkpoint_hidden->bytes()
                           : 0;
    cyclic_n = static_cast<std::size_t>(config_.fingerprint.cyclic_lane_bytes);

    if (use_head) {
        const PrefixReusePath want = reuse;
        auto pick = [&](const DiskCheckpointSlot& s) {
            return s.frontier == reuse_base &&
                   reuse_path_for_context_checkpoint_kind(s.kind) == want &&
                   !checkpoint_missing_hidden(s.hidden_id, config_.hidden_bytes);
        };
        if (pick(meta.rollback)) {
            slot = meta.rollback;
        } else if (pick(meta.ladders[0])) {
            slot = meta.ladders[0];
        } else if (pick(meta.ladders[1])) {
            slot = meta.ladders[1];
        } else {
            restore_failed_ = true;
            ++drops_;
            release_owner();
            return;
        }
        if (hidden_n != 0 && slot.hidden_id == 0) {
            restore_failed_ = true;
            ++drops_;
            release_owner();
            return;
        }
    } else if (checkpoint_missing_hidden(meta.current_hidden_id, config_.hidden_bytes) ||
               (hidden_n != 0 && meta.current_hidden_id == 0)) {
        restore_failed_ = true;
        ++drops_;
        release_owner();
        return;
    }

    lock.unlock();
    maybe_restore_state_barrier();
    restore_io_started = std::chrono::steady_clock::now();

    checkpoint_host.disk_entry_id = entry;
    auto abort_stale = [&] {
        if (failed || stale) { return true; }
        std::lock_guard inner(mutex_);
        if (!restore_state_is_live(epoch, entry, committed, claim, reuse, reuse_base)) {
            stale = true;
        }
        return stale;
    };
    if (!abort_stale()) {
        try {
            checkpoint_prepare = std::jthread([&] {
                try {
                    checkpoint_prepare_ok = prepare_checkpoint_decode(
                        checkpoint_host, checkpoint_plan, epoch, entry, committed, claim, reuse,
                        reuse_base, true);
                } catch (...) { checkpoint_prepare_ok = false; }
                checkpoint_prepare_finished.store(true, std::memory_order_release);
            });
            checkpoint_prepare_started = true;
        } catch (...) {
            checkpoint_prepare_started = false;
        }
        if (checkpoint_prepare_started &&
            fail_after_checkpoint_prepare_start_.exchange(false, std::memory_order_acq_rel)) {
            while (!checkpoint_prepare_barrier_entered_.load(std::memory_order_acquire) &&
                   !checkpoint_prepare_finished.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            lock.lock();
            checkpoint_prepare_barrier_continue_.store(true, std::memory_order_release);
            checkpoint_prepare_barrier_armed_.store(false, std::memory_order_release);
            throw std::runtime_error("injected failure after checkpoint preparation start");
        }
        std::vector<StateDecodeJob> jobs;
        jobs.reserve(6);
        auto enqueue = [&](std::uint64_t id, DiskStateKind kind,
                           std::span<std::uint8_t> capacity,
                           std::span<std::uint8_t>& dst, std::size_t n) {
            if (id == 0 || n == 0) { return; }
            if (n > capacity.size()) {
                throw std::logic_error("KV disk immediate-state arena slice overflow");
            }
            dst = capacity.first(n);
            jobs.push_back(StateDecodeJob{id, kind, dst.data(), n, true});
        };
        auto enqueue_gdn = [&](std::uint64_t id, DiskStateKind kind,
                               std::span<std::uint8_t> conv_capacity,
                               std::span<std::uint8_t> rec_capacity,
                               std::span<std::uint8_t>& conv,
                               std::span<std::uint8_t>& rec) {
            if (id == 0 || conv_n + rec_n == 0) { return; }
            if (conv_n > conv_capacity.size() || rec_n > rec_capacity.size()) {
                throw std::logic_error("KV disk immediate GDN arena slice overflow");
            }
            conv = conv_capacity.first(conv_n);
            rec = rec_capacity.first(rec_n);
            jobs.push_back(StateDecodeJob{id, kind, conv.data(), conv_n, true, rec.data(), rec_n});
        };
        if (use_head) {
            const DiskStateKind gdn_kind = slot.kind == ContextCheckpointKind::TurnRollback
                                                ? DiskStateKind::RollbackGdn
                                                : DiskStateKind::LadderGdn;
            const DiskStateKind hid_kind = slot.kind == ContextCheckpointKind::TurnRollback
                                                ? DiskStateKind::RollbackHidden
                                                : DiskStateKind::LadderHidden;
            const DiskStateKind cyc_kind = slot.kind == ContextCheckpointKind::TurnRollback
                                                ? DiskStateKind::DflashRollback
                                                : DiskStateKind::DflashLadder;
            enqueue_gdn(slot.gdn_id, gdn_kind, restore_state_capacity_.gdn_conv,
                        restore_state_capacity_.gdn_rec, state_slices.gdn_conv,
                        state_slices.gdn_rec);
            enqueue(slot.hidden_id, hid_kind, restore_state_capacity_.hidden,
                    state_slices.hidden, hidden_n);
            enqueue(slot.dflash_id, cyc_kind, restore_state_capacity_.cyclic,
                    state_slices.cyclic, cyclic_n);
        } else {
            enqueue_gdn(meta.current_gdn_id, DiskStateKind::CurrentGdn,
                        restore_state_capacity_.gdn_conv, restore_state_capacity_.gdn_rec,
                        state_slices.gdn_conv, state_slices.gdn_rec);
            enqueue(meta.current_hidden_id, DiskStateKind::TailHidden,
                    restore_state_capacity_.hidden, state_slices.hidden, hidden_n);
            enqueue(meta.current_cyclic_id, DiskStateKind::DflashLocal,
                    restore_state_capacity_.cyclic, state_slices.cyclic, cyclic_n);
        }
        if (unpack_rewrite) {
            if (meta.rewrite_valid &&
                checkpoint_missing_hidden(meta.rewrite_hidden_id, config_.hidden_bytes)) {
                failed = true;
            } else {
                enqueue_gdn(meta.rewrite_gdn_id, DiskStateKind::RewriteGdn,
                            restore_state_capacity_.rewrite_gdn_conv,
                            restore_state_capacity_.rewrite_gdn_rec,
                            state_slices.rewrite_gdn_conv, state_slices.rewrite_gdn_rec);
                enqueue(meta.rewrite_hidden_id, DiskStateKind::RewriteHidden,
                        restore_state_capacity_.rewrite_hidden,
                        state_slices.rewrite_hidden, rewrite_hidden_n);
                enqueue(meta.rewrite_cyclic_id, DiskStateKind::DflashRewrite,
                        restore_state_capacity_.rewrite_cyclic,
                        state_slices.rewrite_cyclic, cyclic_n);
            }
        }
        if (!failed && !decode_state_parallel(jobs)) { failed = true; }
        if (checkpoint_prepare.joinable()) { checkpoint_prepare.join(); }
        if (!failed && !abort_stale()) {
            if (!checkpoint_prepare_started) {
                checkpoint_prepare_ok = prepare_checkpoint_decode(
                    checkpoint_host, checkpoint_plan, epoch, entry, committed, claim, reuse,
                    reuse_base, true);
            }
            if (!checkpoint_prepare_ok ||
                !finish_checkpoint_decode(checkpoint_host, checkpoint_plan, epoch, entry,
                                          committed, claim, reuse, reuse_base, true)) {
                if (!abort_stale()) { failed = true; }
            }
        }
    }
    } catch (...) {
        if (checkpoint_prepare.joinable()) {
            if (lock.owns_lock()) { lock.unlock(); }
            checkpoint_prepare.join();
        }
        bill_restore_io();
        publish_setup_failure();
        return;
    }

    if (!lock.owns_lock()) { lock.lock(); }
    bill_restore_io();
    if (!restore_state_is_live(epoch, entry, committed, claim, reuse, reuse_base) ||
        restore_state_loaded_) {
        release_owner();
        return;
    }
    if (failed || restore_failed_) {
        restore_failed_ = true;
        ++drops_;
        release_owner();
        return;
    }
    restore_state_slices_      = state_slices;
    restore_checkpoint_host_   = std::move(checkpoint_host);
    restore_state_loaded_      = true;
    release_owner();
}

void KVDiskCache::process_job(const Job& job) {
    if (job.kind == JobKind::PrefetchWindow || job.kind == JobKind::RestoreRead) {
        struct ReaderClaimGuard {
            KVDiskCache& cache;
            const Job& job;
            ~ReaderClaimGuard() { cache.drop_reader_claim(job); }
        } claim_guard{*this, job};
        maybe_restore_job_barrier();
        std::uint32_t slot = window_slots();
        {
            std::unique_lock lock(mutex_);
            if (!restore_job_is_live(job)) {
                if (job.kind == JobKind::PrefetchWindow) { unpin_disk(job.disk_entry_id); }
                return;
            }
            for (;;) {
                idle_cv_.wait_for(lock, std::chrono::milliseconds(2), [&] {
                    if (stopping_ || !restore_job_is_live(job)) { return true; }
                    reclaim_completed_h2d_slots();
                    for (std::uint32_t i = 0; i < window_slots(); ++i) {
                        if (!window_[i].assigned) {
                            slot = i;
                            return true;
                        }
                    }
                    return false;
                });
                if (stopping_ || !restore_job_is_live(job)) {
                    if (job.kind == JobKind::PrefetchWindow) { unpin_disk(job.disk_entry_id); }
                    return;
                }
                if (slot < window_slots()) { break; }
            }
            window_[slot].assigned              = true;
            window_[slot].filled                = false;
            window_[slot].h2d_done              = false;
            window_[slot].disk_entry_id        = job.disk_entry_id;
            window_[slot].epoch                 = job.restore_epoch;
            window_[slot].claim_generation      = job.claim_generation;
            window_[slot].committed_generation   = job.committed_generation;
            window_[slot].pool                  = job.pool;
            window_[slot].logical_index         = job.logical_index;
            ++window_inflight_;
        }
        if (job.logical_index != 0) { maybe_slot_assign_barrier(job.logical_index); }
        std::uint64_t object_id = 0;
        DiskObjectKind kind     = DiskObjectKind::Main;
        {
            std::lock_guard lock(mutex_);
            if (!restore_job_is_live(job)) {
                if (window_inflight_ > 0) { --window_inflight_; }
                reset_window_slot_keep_host(window_[slot]);
                if (job.kind == JobKind::PrefetchWindow) { unpin_disk(job.disk_entry_id); }
                idle_cv_.notify_all();
                return;
            }
            const auto it = entries_.find(job.disk_entry_id != 0 ? job.disk_entry_id
                                                                 : restore_entry_id_);
            if (it == entries_.end()) {
                --window_inflight_;
                if (restore_job_is_live(job)) { publish_page_job_failure(job); }
                reset_window_slot_keep_host(window_[slot]);
                if (job.kind == JobKind::PrefetchWindow) { unpin_disk(job.disk_entry_id); }
                idle_cv_.notify_all();
                return;
            }
            kind      = job.pool == 0 ? DiskObjectKind::Main : DiskObjectKind::Backend;
            const auto& ids =
                job.pool == 0 ? it->second.meta.main_page_ids : it->second.meta.backend_page_ids;
            if (job.logical_index >= ids.size()) {
                --window_inflight_;
                if (restore_job_is_live(job)) { publish_page_job_failure(job); }
                reset_window_slot_keep_host(window_[slot]);
                if (job.kind == JobKind::PrefetchWindow) { unpin_disk(job.disk_entry_id); }
                idle_cv_.notify_all();
                return;
            }
            object_id = ids[job.logical_index];
        }
        try {
            fill_window_slot(slot, kind, object_id, job.pool, job.logical_index);
        } catch (...) {
            finish_window_inflight(slot, false);
            std::lock_guard lock(mutex_);
            if (restore_job_is_live(job)) { publish_page_job_failure(job); }
            if (job.kind == JobKind::PrefetchWindow) { unpin_disk(job.disk_entry_id); }
            return;
        }
        finish_window_inflight(slot, true);
        if (job.kind == JobKind::PrefetchWindow) {
            std::lock_guard lock(mutex_);
            unpin_disk(job.disk_entry_id);
        }
        if (job.kind == JobKind::RestoreRead && restore_io_threads_ == 1) {
            std::unique_lock lock(mutex_);
            if (restore_job_is_live(job) && restore_q_.empty() && emergency_q_.empty()) {
                bool pending_slot = false;
                for (const WindowSlot& win : window_) {
                    if (win.assigned && !win.filled && !win.h2d_done) { pending_slot = true; }
                }
                if (!pending_slot && !restore_state_loaded_ && !restore_failed_) {
                    load_restore_state_locked(lock);
                }
            }
        }
        return;
    }

    if (job.kind == JobKind::EmergencySpillPage || job.kind == JobKind::IdleSpillPage ||
        job.kind == JobKind::EmergencyCommit || job.kind == JobKind::IdleCommit) {
        maybe_payload_take_barrier();
        if (fail_after_payload_take_.exchange(false, std::memory_order_acq_rel)) {
            throw std::runtime_error("injected payload-take failure");
        }
    }

    if (job.kind == JobKind::EmergencySpillPage || job.kind == JobKind::IdleSpillPage) {
        std::vector<Job> batch;
        std::unique_lock lock(mutex_);
        if (!spill_ || spill_->ram_id != job.ram_entry_id || spill_->epoch != job.spill_epoch) {
            finish_payload_io(lock);
            return;
        }
        if (spill_->cancelled) {
            finish_payload_io(lock);
            return;
        }
        batch.push_back(job);
        const std::size_t batch_limit = std::clamp<std::size_t>(
            config_.pack_page_batch == 0 ? 1 : config_.pack_page_batch, 1, 8);
        auto& queued = job.kind == JobKind::EmergencySpillPage ? emergency_q_ : idle_q_;
        while (batch.size() < batch_limit && !queued.empty()) {
            const Job& next = queued.front();
            if (next.kind != job.kind || next.ram_entry_id != job.ram_entry_id ||
                next.spill_epoch != job.spill_epoch || next.pool != job.pool ||
                next.logical_index != job.logical_index + batch.size()) {
                break;
            }
            batch.push_back(next);
            queued.pop_front();
            ++payload_io_inflight_;
        }
        try {
            spill_page_batch(*spill_, batch, lock);
        } catch (...) {
            for (std::size_t i = 0; i < batch.size(); ++i) { finish_payload_io(lock); }
            if (spill_ && !spill_->committed && !(spill_->meta_installed)) {
                drop_spill(*spill_, lock);
            }
            spill_.reset();
            idle_cv_.notify_all();
            cv_.notify_all();
            return;
        }
        for (std::size_t i = 0; i < batch.size(); ++i) { finish_payload_io(lock); }
        return;
    }

    if (job.kind == JobKind::EmergencyCommit || job.kind == JobKind::IdleCommit) {
        std::unique_lock lock(mutex_);
        if (!spill_ || spill_->ram_id != job.ram_entry_id || spill_->epoch != job.spill_epoch) {
            finish_payload_io(lock);
            return;
        }
        if (job.kind == JobKind::IdleCommit && !spill_->emergency && spill_->cancelled) {
            finish_payload_io(lock);
            return;
        }
        try {
            commit_spill(*spill_, lock);
        } catch (...) {
            finish_payload_io(lock);
            if (spill_ && !spill_->committed && !(spill_->meta_installed)) {
                drop_spill(*spill_, lock);
            }
            spill_.reset();
            idle_cv_.notify_all();
            cv_.notify_all();
            return;
        }
        finish_payload_io(lock);
        if (spill_) { purge_spill_jobs(spill_->epoch); }
        spill_.reset();
        idle_cv_.notify_all();
        cv_.notify_all();
    }
}

void KVDiskCache::join_io_threads() noexcept {
    for (std::thread& t : restore_threads_) {
        if (t.joinable()) { t.join(); }
    }
    restore_threads_.clear();
    if (io_thread_.joinable()) { io_thread_.join(); }
}

void KVDiskCache::stop_io_threads() noexcept {
    if (!io_thread_.joinable() && restore_threads_.empty()) { return; }
    try {
        {
            std::lock_guard lock(mutex_);
            restore_job_continue_.store(true, std::memory_order_release);
            restore_state_continue_.store(true, std::memory_order_release);
            restore_job_barrier_armed_.store(false, std::memory_order_release);
            restore_state_barrier_armed_.store(false, std::memory_order_release);
            copy_lease_continue_.store(true, std::memory_order_release);
            copy_lease_barrier_armed_.store(false, std::memory_order_release);
            wait_epoch_continue_.store(true, std::memory_order_release);
            wait_epoch_barrier_armed_.store(false, std::memory_order_release);
            page_read_continue_.store(true, std::memory_order_release);
            page_read_barrier_armed_.store(false, std::memory_order_release);
            direct_state_read_continue_.store(true, std::memory_order_release);
            direct_state_read_barrier_armed_.store(false, std::memory_order_release);
            payload_take_continue_.store(true, std::memory_order_release);
            payload_take_barrier_armed_.store(false, std::memory_order_release);
            slot_assign_continue_.store(true, std::memory_order_release);
            slot_assign_barrier_armed_.store(false, std::memory_order_release);
            scatter_record_continue_.store(true, std::memory_order_release);
            scatter_record_barrier_armed_.store(false, std::memory_order_release);
        }
        try {
            cancel_restore();
        } catch (...) {}
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        idle_cv_.notify_all();
        join_io_threads();
    } catch (...) {}
}

void KVDiskCache::restore_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] {
                return (stopping_ && restore_q_.empty() && prefetch_q_.empty()) ||
                       (!restore_q_.empty() && !emergency_payload_busy_locked()) ||
                       prefetch_readable_locked() ||
                       (restore_active_ && restore_target_ && !restore_state_loaded_ &&
                        !restore_failed_ && restore_state_inflight_ == 0 &&
                        !emergency_payload_busy_locked());
            });
            if (stopping_ && restore_q_.empty() && prefetch_q_.empty()) { return; }
            if (!stopping_ && restore_active_ && restore_target_ && !restore_state_loaded_ &&
                !restore_failed_ && !emergency_payload_busy_locked() &&
                restore_state_inflight_ == 0) {
                try {
                    load_restore_state_locked(lock);
                } catch (...) {
                    if (!lock.owns_lock()) { lock.lock(); }
                    if (restore_state_inflight_ > 0) { --restore_state_inflight_; }
                    if (restore_active_ && restore_target_ && !restore_state_loaded_ &&
                        !restore_failed_) {
                        restore_failed_ = true;
                        ++drops_;
                    }
                    idle_cv_.notify_all();
                }
                continue;
            }
            if (auto taken = take_restore_job(lock)) {
                job = *taken;
            } else {
                if (!stopping_) {
                    restore_loop_idle_spins_.fetch_add(1, std::memory_order_relaxed);
                }
                continue;
            }
        }
        try {
            process_job(job);
        } catch (...) {
            std::unique_lock lock(mutex_);
            ++drops_;
            if (page_job_is_live_restore(job)) { restore_failed_ = true; }
            if (job.kind == JobKind::PrefetchWindow) {
                try {
                    unpin_disk(job.disk_entry_id);
                } catch (...) {}
            }
            idle_cv_.notify_all();
        }
    }
}

void KVDiskCache::io_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] {
                if (stopping_) {
                    if (emergency_q_.empty() && idle_q_.empty() &&
                        (restore_io_threads_ > 1 ||
                         (restore_q_.empty() && prefetch_q_.empty()))) {
                        return true;
                    }
                    return !restore_readers_busy_locked();
                }
                if (restore_io_threads_ == 1 &&
                    ((!restore_q_.empty() && !emergency_payload_busy_locked()) ||
                     prefetch_readable_locked())) {
                    return true;
                }
                if (!emergency_q_.empty() && !restore_readers_busy_locked()) { return true; }
                if ((!idle_q_.empty() || idle_requested_) && !restore_or_prefetch_busy_locked()) {
                    return true;
                }
                return restore_active_ && restore_target_ && !restore_state_loaded_ &&
                       !restore_failed_ && restore_state_inflight_ == 0;
            });
            if (stopping_) {
                if (restore_io_threads_ > 1) {
                    if (emergency_q_.empty() && idle_q_.empty()) { return; }
                } else if (restore_q_.empty() && emergency_q_.empty() && idle_q_.empty() &&
                           prefetch_q_.empty()) {
                    return;
                }
            }
            if (auto taken = take_job(lock)) {
                job = *taken;
            } else if (!stopping_ && restore_active_ && restore_target_ && !restore_state_loaded_ &&
                       !restore_failed_ && restore_state_inflight_ == 0) {
                try {
                    load_restore_state_locked(lock);
                } catch (...) {
                    if (!lock.owns_lock()) { lock.lock(); }
                    if (restore_state_inflight_ > 0) { --restore_state_inflight_; }
                    if (restore_active_ && restore_target_ && !restore_state_loaded_ &&
                        !restore_failed_) {
                        restore_failed_ = true;
                        ++drops_;
                    }
                    idle_cv_.notify_all();
                }
                continue;
            } else if (!stopping_ && idle_requested_ && !spill_ && !idle_cancel_all_ &&
                       !restore_or_prefetch_busy_locked() && config_.ram != nullptr) {
                std::uint64_t ram_id = 0;
                for (const std::uint64_t id : config_.ram->fifo_ids()) {
                    if (idle_cancel_rams_.count(id) != 0) { continue; }
                    if (config_.ram->is_claimed(id)) { continue; }
                    const auto it = ram_notes_.find(id);
                    if (it != ram_notes_.end() && !it->second.durable &&
                        !(it->second.failed_this_generation &&
                          it->second.generation_stamp == durable_generation_)) {
                        ram_id = id;
                        break;
                    }
                }
                if (ram_id == 0) {
                    idle_requested_ = false;
                    idle_cv_.notify_all();
                    continue;
                }
                const std::uint64_t epoch = idle_cancel_epoch_;
                idle_pinning_             = true;
                idle_pinning_ram_         = ram_id;
                lock.unlock();
                bool ram_pinned = false;
                try {
                    config_.ram->pin_for_io(ram_id);
                    ram_pinned = true;
                    while (!config_.ram->copies_ready(ram_id)) { config_.ram->wait_pending_copies(); }
                } catch (...) {
                    if (ram_pinned) {
                        try {
                            config_.ram->unpin_for_io(ram_id);
                        } catch (...) {}
                    }
                    std::lock_guard inner(mutex_);
                    idle_pinning_     = false;
                    idle_pinning_ram_ = 0;
                    idle_cv_.notify_all();
                    auto it = ram_notes_.find(ram_id);
                    if (it != ram_notes_.end()) {
                        it->second.failed_this_generation = true;
                        it->second.generation_stamp       = durable_generation_;
                    }
                    continue;
                }
                std::unique_lock inner(mutex_);
                const bool abort_pin = epoch != idle_cancel_epoch_ || idle_cancel_all_ ||
                                       idle_cancel_rams_.count(ram_id) != 0 ||
                                       config_.ram->is_claimed(ram_id) ||
                                       restore_cancels_idle_locked();
                bool prepared = false;
                try {
                    prepared = !abort_pin && prepare_spill(ram_id, false, inner);
                } catch (...) {
                    prepared = false;
                    auto it = ram_notes_.find(ram_id);
                    if (it != ram_notes_.end()) {
                        it->second.failed_this_generation = true;
                        it->second.generation_stamp       = durable_generation_;
                    }
                    ++drops_;
                }
                if (abort_pin || !prepared) {
                    try {
                        config_.ram->unpin_for_io(ram_id);
                    } catch (...) {}
                    idle_pinning_     = false;
                    idle_pinning_ram_ = 0;
                    idle_cv_.notify_all();
                    continue;
                }
                idle_pinning_     = false;
                idle_pinning_ram_ = 0;
                idle_cv_.notify_all();
                continue;
            } else {
                continue;
            }
        }
        try {
            process_job(job);
        } catch (...) {
            std::unique_lock lock(mutex_);
            if (job.kind == JobKind::EmergencySpillPage || job.kind == JobKind::IdleSpillPage ||
                job.kind == JobKind::EmergencyCommit || job.kind == JobKind::IdleCommit) {
                finish_payload_io(lock);
            }
            ++drops_;
            if (page_job_is_live_restore(job)) { restore_failed_ = true; }
            if (job.kind == JobKind::PrefetchWindow) {
                try {
                    unpin_disk(job.disk_entry_id);
                } catch (...) {}
            }
            if (spill_ && !spill_->committed && !spill_->failed) {
                drop_spill(*spill_, lock);
                spill_.reset();
            }
            idle_cv_.notify_all();
            cv_.notify_all();
        }
    }
}

std::uint32_t KVDiskCache::test_crc32c(std::span<const std::uint8_t> bytes) noexcept {
    return crc32c(bytes);
}

void KVDiskCache::test_break_object(std::uint64_t object_id, DiskObjectKind kind) {
    std::lock_guard lock(mutex_);
    const auto it = objects_.find(object_id);
    if (it == objects_.end() || it->second.kind != kind) { return; }
    const auto path = pack_path(*it->second.location.generation, kind,
                                it->second.location.segment);
    const auto offset = it->second.location.offset;
    std::uint8_t zero[8]{};
    const int fd = ::open(path.c_str(), O_WRONLY);
    if (fd < 0) { return; }
    const ssize_t written = ::pwrite(fd, zero, sizeof(zero), static_cast<off_t>(offset));
    if (written != static_cast<ssize_t>(sizeof(zero))) {
        (void)::close(fd);
        throw std::runtime_error("failed to corrupt KV disk test object");
    }
    (void)::fsync(fd);
    ::close(fd);
}

std::uint64_t KVDiskCache::test_object_refcount(std::uint64_t object_id) const {
    std::lock_guard lock(mutex_);
    const auto it = objects_.find(object_id);
    return it == objects_.end() ? 0 : it->second.live_refs + it->second.skip_refs;
}

bool KVDiskCache::test_entry_in_index(std::uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    return entries_.count(entry_id) != 0;
}

std::vector<std::uint64_t> KVDiskCache::test_main_page_ids(std::uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    return require(entry_id).meta.main_page_ids;
}

std::vector<std::uint64_t> KVDiskCache::test_backend_page_ids(std::uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    return require(entry_id).meta.backend_page_ids;
}

DiskMeta KVDiskCache::test_load_meta(std::uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    return require(entry_id).meta;
}

std::size_t KVDiskCache::test_skipped_count() const {
    std::lock_guard lock(mutex_);
    return skipped_.size();
}

void KVDiskCache::test_arm_crash_before_meta() {
    std::lock_guard lock(mutex_);
    crash_before_meta_ = true;
}

void KVDiskCache::test_arm_stall_after_meta_rename() {
    meta_renamed_.store(false, std::memory_order_release);
    stall_after_meta_rename_ = true;
}

void KVDiskCache::test_arm_fail_after_meta_rename() {
    meta_renamed_.store(false, std::memory_order_release);
    fail_after_meta_rename_ = true;
}

void KVDiskCache::test_arm_fail_rollback_meta() {
    fail_rollback_meta_ = true;
}

void KVDiskCache::test_arm_fail_after_rollback_rename() {
    fail_after_rollback_rename_ = true;
}

void KVDiskCache::test_arm_fail_tombstone() {
    fail_tombstone_ = true;
}

void KVDiskCache::test_arm_fail_object_write() {
    fail_object_write_ = true;
}

void KVDiskCache::test_arm_fail_object_unlink() {
    fail_object_unlink_ = true;
}

void KVDiskCache::test_arm_fail_prepare_spill() {
    fail_prepare_spill_ = true;
}

bool KVDiskCache::test_fifo_evict_one_unpersisted() {
    std::unique_lock lock(mutex_);
    const std::size_t before = fifo_.size();
    fifo_evict_one();
    (void)flush_queued_unlinks(lock);
    return fifo_.size() < before;
}

bool KVDiskCache::test_gc_skipped_one() {
    std::unique_lock lock(mutex_);
    const bool ok = gc_skipped_one();
    const bool unlinked = flush_queued_unlinks(lock);
    if (ok && unlinked) { persist_eviction(lock); }
    return ok;
}

bool KVDiskCache::test_fifo_evict_one() {
    std::unique_lock lock(mutex_);
    const std::size_t before = fifo_.size();
    fifo_evict_one();
    const bool unlinked = flush_queued_unlinks(lock);
    if (fifo_.size() < before && unlinked) { persist_eviction(lock); }
    return fifo_.size() < before;
}

bool KVDiskCache::test_meta_renamed() const {
    return meta_renamed_.load(std::memory_order_acquire);
}

bool KVDiskCache::test_objects_fsynced_before_meta() const {
    return objects_dir_fsynced_.load(std::memory_order_acquire);
}

std::uint32_t KVDiskCache::test_disk_io_pins(std::uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(entry_id);
    return it == entries_.end() ? 0 : it->second.io_pins;
}

void KVDiskCache::test_force_zstd_fail() {
    std::lock_guard lock(mutex_);
    force_zstd_fail_ = true;
}

void KVDiskCache::test_set_payload_io_stall_ms(int ms) {
    payload_io_entered_.store(false, std::memory_order_release);
    payload_io_stall_ms_.store(ms, std::memory_order_release);
}

bool KVDiskCache::test_payload_io_entered() const {
    return payload_io_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_set_manifest_io_stall_ms(int ms) {
    manifest_io_entered_.store(false, std::memory_order_release);
    manifest_io_stall_ms_.store(ms, std::memory_order_release);
}

bool KVDiskCache::test_manifest_io_entered() const {
    return manifest_io_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_set_free_bytes_override(std::optional<std::uint64_t> bytes) {
    std::lock_guard lock(mutex_);
    test_free_bytes_override_ = bytes;
}

void KVDiskCache::test_set_pack_position(DiskObjectKind kind, std::uint32_t segment,
                                         std::uint64_t tail) {
    if (kind_index(kind) >= pack_active_tail_.size() || tail > kPackSegmentBytes ||
        tail % kDiskPageIoAlignment != 0) {
        throw std::invalid_argument("invalid test KV pack position");
    }
    std::lock_guard lock(mutex_);
    if (spill_ || restore_active_ || restore_target_ || payload_io_inflight_ != 0 ||
        window_inflight_ != 0 || restore_state_inflight_ != 0) {
        throw std::logic_error("cannot move test KV pack position during I/O");
    }
    const std::size_t index = kind_index(kind);
    pack_active_segment_[index] = segment;
    pack_active_tail_[index] = tail;
}

void KVDiskCache::test_arm_partial_pwritev(std::size_t max_bytes) {
    if (max_bytes == 0) { throw std::invalid_argument("partial pwritev limit must be nonzero"); }
    test_partial_pwritev_bytes_.store(max_bytes, std::memory_order_release);
}

std::size_t KVDiskCache::take_test_partial_pwritev_bytes() noexcept {
    return test_partial_pwritev_bytes_.exchange(0, std::memory_order_acq_rel);
}

void KVDiskCache::test_arm_fault(DiskFaultPoint point) {
    test_next_fault_point_.store(DiskFaultPoint::None, std::memory_order_release);
    test_fault_point_.store(point, std::memory_order_release);
}

void KVDiskCache::test_arm_fault_sequence(DiskFaultPoint first, DiskFaultPoint second) {
    if (first == DiskFaultPoint::None || second == DiskFaultPoint::None) {
        throw std::invalid_argument("fault sequence points must be nonempty");
    }
    test_next_fault_point_.store(second, std::memory_order_release);
    test_fault_point_.store(first, std::memory_order_release);
}

void KVDiskCache::maybe_test_fault(DiskFaultPoint point) const {
    DiskFaultPoint expected = point;
    if (test_fault_point_.compare_exchange_strong(expected, DiskFaultPoint::None,
                                                  std::memory_order_acq_rel)) {
        test_fault_point_.store(
            test_next_fault_point_.exchange(DiskFaultPoint::None, std::memory_order_acq_rel),
            std::memory_order_release);
        throw std::runtime_error("injected KV disk durability phase failure");
    }
}

std::shared_ptr<void> KVDiskCache::test_hold_active_generation() const {
    std::lock_guard lock(mutex_);
    return active_generation_;
}

void KVDiskCache::test_set_fsync_stall_ms(int ms) {
    fsync_entered_.store(false, std::memory_order_release);
    fsync_stall_ms_.store(ms, std::memory_order_release);
}

bool KVDiskCache::test_fsync_entered() const {
    return fsync_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_arm_restore_job_barrier() {
    restore_job_dequeued_.store(false, std::memory_order_release);
    restore_job_continue_.store(false, std::memory_order_release);
    restore_job_barrier_armed_.store(true, std::memory_order_release);
}

bool KVDiskCache::test_restore_job_dequeued() const {
    return restore_job_dequeued_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_restore_job_barrier() {
    restore_job_continue_.store(true, std::memory_order_release);
    restore_job_barrier_armed_.store(false, std::memory_order_release);
}

void KVDiskCache::test_arm_restore_state_barrier() {
    restore_state_entered_.store(false, std::memory_order_release);
    restore_state_continue_.store(false, std::memory_order_release);
    restore_state_barrier_armed_.store(true, std::memory_order_release);
}

bool KVDiskCache::test_restore_state_entered() const {
    return restore_state_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_restore_state_barrier() {
    restore_state_continue_.store(true, std::memory_order_release);
    restore_state_barrier_armed_.store(false, std::memory_order_release);
}

bool KVDiskCache::test_has_ram_note(std::uint64_t ram_id) const {
    std::lock_guard lock(mutex_);
    return ram_notes_.count(ram_id) != 0;
}

void KVDiskCache::test_arm_copy_lease_barrier() {
    copy_event_leased_.store(false, std::memory_order_release);
    copy_lease_continue_.store(false, std::memory_order_release);
    copy_lease_barrier_armed_.store(true, std::memory_order_release);
}

bool KVDiskCache::test_copy_event_leased() const {
    return copy_event_leased_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_copy_lease_barrier() {
    copy_lease_continue_.store(true, std::memory_order_release);
    copy_lease_barrier_armed_.store(false, std::memory_order_release);
}

cudaEvent_t KVDiskCache::test_copies_done() const {
    std::lock_guard lock(mutex_);
    return copies_done_;
}

std::size_t KVDiskCache::test_retired_copy_events() const {
    std::lock_guard lock(mutex_);
    return retired_copy_events_.size();
}

void KVDiskCache::test_arm_fail_page_read() {
    fail_next_page_read_.store(true, std::memory_order_release);
}

void KVDiskCache::test_arm_fail_restore_state_setup() {
    fail_next_restore_state_setup_.store(true, std::memory_order_release);
}

void KVDiskCache::test_arm_fail_after_checkpoint_prepare_start() {
    checkpoint_prepare_barrier_entered_.store(false, std::memory_order_release);
    checkpoint_prepare_barrier_continue_.store(false, std::memory_order_release);
    checkpoint_prepare_barrier_armed_.store(true, std::memory_order_release);
    fail_after_checkpoint_prepare_start_.store(true, std::memory_order_release);
}

void KVDiskCache::test_arm_fail_after_state_h2d_enqueue() {
    fail_after_state_h2d_enqueue_.store(true, std::memory_order_release);
}

void KVDiskCache::test_arm_direct_state_read_barrier() {
    direct_state_read_entered_.store(false, std::memory_order_release);
    direct_state_read_continue_.store(false, std::memory_order_release);
    direct_state_read_barrier_armed_.store(true, std::memory_order_release);
}

bool KVDiskCache::test_direct_state_read_entered() const {
    return direct_state_read_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_direct_state_read_barrier() {
    direct_state_read_continue_.store(true, std::memory_order_release);
    direct_state_read_barrier_armed_.store(false, std::memory_order_release);
}

void KVDiskCache::test_arm_fail_wait_copies() {
    fail_next_wait_copies_.store(true, std::memory_order_release);
}

void KVDiskCache::test_set_state_decode_stall_ms(int ms) {
    state_decode_count_.store(0, std::memory_order_release);
    state_decode_stall_ms_.store(ms, std::memory_order_release);
}

int KVDiskCache::test_state_decode_count() const {
    return state_decode_count_.load(std::memory_order_acquire);
}

void KVDiskCache::test_arm_wait_epoch_barrier() {
    wait_epoch_latched_.store(false, std::memory_order_release);
    wait_epoch_continue_.store(false, std::memory_order_release);
    wait_epoch_barrier_armed_.store(true, std::memory_order_release);
}

bool KVDiskCache::test_wait_epoch_latched() const {
    return wait_epoch_latched_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_wait_epoch_barrier() {
    wait_epoch_continue_.store(true, std::memory_order_release);
    wait_epoch_barrier_armed_.store(false, std::memory_order_release);
}

cudaEvent_t KVDiskCache::test_last_wait_copy_event() const {
    std::lock_guard lock(mutex_);
    return last_wait_copy_event_;
}

std::uint32_t KVDiskCache::test_timing_harvests() const {
    std::lock_guard lock(mutex_);
    return timing_harvests_;
}

std::uint64_t KVDiskCache::test_committed_generation(std::uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(entry_id);
    return it == entries_.end() ? 0 : it->second.committed_generation;
}

bool KVDiskCache::test_window_filled_for(std::uint64_t entry_id) const {
    std::lock_guard lock(mutex_);
    for (const WindowSlot& slot : window_) {
        if (slot.disk_entry_id == entry_id && slot.filled) { return true; }
    }
    return false;
}

void KVDiskCache::test_arm_page_read_barrier() {
    page_read_count_.store(0, std::memory_order_release);
    page_read_entered_.store(false, std::memory_order_release);
    page_read_continue_.store(false, std::memory_order_release);
    page_read_barrier_armed_.store(true, std::memory_order_release);
}

bool KVDiskCache::test_page_read_entered() const {
    return page_read_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_page_read_barrier() {
    page_read_continue_.store(true, std::memory_order_release);
    page_read_barrier_armed_.store(false, std::memory_order_release);
}

std::uint32_t KVDiskCache::test_restore_state_inflight() const {
    std::lock_guard lock(mutex_);
    return restore_state_inflight_;
}

std::uint32_t KVDiskCache::test_window_inflight() const {
    std::lock_guard lock(mutex_);
    return window_inflight_;
}

int KVDiskCache::test_page_read_count() const {
    return page_read_count_.load(std::memory_order_acquire);
}

void KVDiskCache::test_reset_restore_loop_idle_spins() {
    restore_loop_idle_spins_.store(0, std::memory_order_release);
}

std::uint64_t KVDiskCache::test_restore_loop_idle_spins() const {
    return restore_loop_idle_spins_.load(std::memory_order_acquire);
}

void KVDiskCache::test_arm_payload_take_barrier() {
    payload_take_entered_.store(false, std::memory_order_release);
    payload_take_continue_.store(false, std::memory_order_release);
    payload_take_barrier_armed_.store(true, std::memory_order_release);
}

bool KVDiskCache::test_payload_take_entered() const {
    return payload_take_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_payload_take_barrier() {
    payload_take_continue_.store(true, std::memory_order_release);
    payload_take_barrier_armed_.store(false, std::memory_order_release);
}

void KVDiskCache::test_arm_slot_assign_barrier() {
    slot_assign_entered_.store(false, std::memory_order_release);
    slot_assign_continue_.store(false, std::memory_order_release);
    slot_assign_barrier_armed_.store(true, std::memory_order_release);
}

bool KVDiskCache::test_slot_assign_entered() const {
    return slot_assign_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_slot_assign_barrier() {
    slot_assign_continue_.store(true, std::memory_order_release);
    slot_assign_barrier_armed_.store(false, std::memory_order_release);
}

void KVDiskCache::test_arm_scatter_record_barrier(std::uint32_t logical_index) {
    if (scatter_test_gate_stream_ == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&scatter_test_gate_stream_, cudaStreamNonBlocking));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(scatter_test_gate_stream_));
    }
    if (scatter_test_gate_ == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&scatter_test_gate_, cudaEventDisableTiming));
    }
    scatter_record_logical_.store(logical_index, std::memory_order_release);
    scatter_record_entered_.store(false, std::memory_order_release);
    scatter_record_continue_.store(false, std::memory_order_release);
    scatter_record_barrier_armed_.store(true, std::memory_order_release);
    try {
        CUDA_CHECK(cudaLaunchHostFunc(
            scatter_test_gate_stream_,
            [](void* opaque) {
                auto* cache = static_cast<KVDiskCache*>(opaque);
                while (cache->scatter_record_barrier_armed_.load(std::memory_order_acquire) &&
                       !cache->scatter_record_continue_.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            },
            this));
        CUDA_CHECK(cudaEventRecord(scatter_test_gate_, scatter_test_gate_stream_));
    } catch (...) {
        scatter_record_continue_.store(true, std::memory_order_release);
        scatter_record_barrier_armed_.store(false, std::memory_order_release);
        throw;
    }
}

bool KVDiskCache::test_scatter_record_entered() const {
    return scatter_record_entered_.load(std::memory_order_acquire);
}

void KVDiskCache::test_release_scatter_record_barrier() {
    scatter_record_continue_.store(true, std::memory_order_release);
    scatter_record_barrier_armed_.store(false, std::memory_order_release);
}

bool KVDiskCache::test_restore_read_queued(std::uint32_t pool, std::uint32_t logical) const {
    std::lock_guard lock(mutex_);
    for (const Job& job : restore_q_) {
        if (job.kind == JobKind::RestoreRead && job.pool == pool && job.logical_index == logical) {
            return true;
        }
    }
    return false;
}

bool KVDiskCache::test_reader_claim(std::uint32_t pool, std::uint32_t logical) const {
    std::lock_guard lock(mutex_);
    for (const ReaderClaim& claim : reader_claims_) {
        if (claim.pool == pool && claim.logical_index == logical) { return true; }
    }
    return false;
}

bool KVDiskCache::test_window_assigned(std::uint32_t pool, std::uint32_t logical) const {
    std::lock_guard lock(mutex_);
    for (const WindowSlot& slot : window_) {
        if (slot.assigned && slot.pool == pool && slot.logical_index == logical) { return true; }
    }
    return false;
}

void KVDiskCache::test_arm_fail_after_payload_take() {
    fail_after_payload_take_.store(true, std::memory_order_release);
}

std::uint32_t KVDiskCache::test_payload_io_inflight() const {
    std::lock_guard lock(mutex_);
    return payload_io_inflight_;
}

std::uint64_t KVDiskCache::test_prefetch_preempted_idle() const {
    return prefetch_preempted_idle_.load(std::memory_order_acquire);
}

bool KVDiskCache::test_emergency_queued() const {
    std::lock_guard lock(mutex_);
    return !emergency_q_.empty();
}

void KVDiskCache::test_stop_io_threads() { stop_io_threads(); }

void KVDiskCache::test_set_stopping() {
    std::lock_guard lock(mutex_);
    stopping_ = true;
    cv_.notify_all();
    idle_cv_.notify_all();
}

} // namespace ninfer::targets::qwen3_6::detail
