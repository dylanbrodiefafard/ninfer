#include "targets/qwen4/verifier.h"

#include "core/device.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;
namespace ops = ninfer::ops;

namespace {

constexpr std::int32_t kVocabulary = 248320;
constexpr std::int32_t kEos = verifier::kPleResetToken;
constexpr std::size_t kCopyChunkBytes = 1U << 20;
constexpr std::array<std::int32_t, 11> kProbePositions = {
    3, 4, 5, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 4095};
constexpr std::array<std::uint64_t, 3> kNgramMultiplier = {
    23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
constexpr std::array<std::int32_t, 16> kNgramPrime = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr std::array<std::int32_t, 16> kNgramOffset = {
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
    300001275};

struct Digest128 {
    std::uint64_t lo = 1469598103934665603ULL;
    std::uint64_t hi = 1099511628211ULL ^ 0xd6e8feb86659fd93ULL;
    std::uint64_t bytes = 0;

    void add(const void* source, std::size_t count) {
        const auto* data = static_cast<const std::uint8_t*>(source);
        std::size_t offset = 0;
        while (offset + sizeof(std::uint64_t) <= count) {
            std::uint64_t word = 0;
            std::memcpy(&word, data + offset, sizeof(word));
            lo = std::rotl((lo ^ word) * 1099511628211ULL, 17);
            hi = std::rotl(hi + (word ^ 0x9e3779b97f4a7c15ULL), 31) *
                 0xbf58476d1ce4e5b9ULL;
            offset += sizeof(word);
        }
        std::uint64_t tail = static_cast<std::uint64_t>(count - offset) << 56U;
        for (std::size_t byte = 0; offset + byte < count; ++byte) {
            tail |= static_cast<std::uint64_t>(data[offset + byte]) << (8U * byte);
        }
        if (offset != count) {
            lo = std::rotl((lo ^ tail) * 1099511628211ULL, 17);
            hi = std::rotl(hi + (tail ^ 0x9e3779b97f4a7c15ULL), 31) *
                 0xbf58476d1ce4e5b9ULL;
        }
        bytes += count;
    }

    template <typename T>
    void add_scalar(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        add(&value, sizeof(value));
    }

    void add_field(std::string_view label, const void* source, std::size_t count) {
        add_scalar(label.size());
        add(label.data(), label.size());
        add_scalar(count);
        add(source, count);
    }

    void add_digest(const Digest128& digest) {
        add_scalar(digest.lo);
        add_scalar(digest.hi);
        add_scalar(digest.bytes);
    }

    bool operator==(const Digest128&) const = default;
};

std::string digest_string(const Digest128& digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << digest.lo << std::setw(16)
        << digest.hi << ':' << std::dec << digest.bytes;
    return out.str();
}

enum class ValidationKind {
    None,
    Bf16,
    Fp32,
    Fp8E4m3,
};

bool finite_bf16(std::uint16_t bits) { return (bits & 0x7f80U) != 0x7f80U; }

void validate_host_bytes(std::span<const std::uint8_t> bytes, ValidationKind kind,
                         std::string_view label, int& failures) {
    bool valid = true;
    if (kind == ValidationKind::Bf16) {
        for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(std::uint16_t)) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
            valid &= finite_bf16(bits);
        }
    } else if (kind == ValidationKind::Fp32) {
        for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
            float value = 0.0F;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            valid &= std::isfinite(value);
        }
    } else if (kind == ValidationKind::Fp8E4m3) {
        for (std::uint8_t bits : bytes) {
            __nv_fp8_e4m3 represented;
            represented.__x = bits;
            valid &= std::isfinite(static_cast<float>(represented));
        }
    }
    if (!valid) {
        std::cerr << "non-finite represented value in " << label << '\n';
        ++failures;
    }
}

void hash_device_tensor(Digest128& digest, const ninfer::Tensor& tensor,
                        std::string_view label, ValidationKind validation,
                        std::vector<std::uint8_t>& scratch, int& failures,
                        std::size_t* byte_counter = nullptr) {
    digest.add_scalar(label.size());
    digest.add(label.data(), label.size());
    digest.add_scalar(tensor.dtype);
    for (std::int32_t extent : tensor.ne) { digest.add_scalar(extent); }
    const std::size_t bytes = tensor.bytes();
    digest.add_scalar(bytes);
    if (byte_counter != nullptr) { *byte_counter += bytes; }
    const std::size_t alignment = validation == ValidationKind::Fp32 ? sizeof(float)
                                  : validation == ValidationKind::Bf16
                                      ? sizeof(std::uint16_t)
                                      : 1U;
    for (std::size_t offset = 0; offset < bytes;) {
        std::size_t count = std::min(scratch.size(), bytes - offset);
        count -= count % alignment;
        if (count == 0) { count = bytes - offset; }
        CUDA_CHECK(cudaMemcpy(scratch.data(), static_cast<const std::byte*>(tensor.data) + offset,
                              count, cudaMemcpyDeviceToHost));
        const std::span<const std::uint8_t> copied(scratch.data(), count);
        validate_host_bytes(copied, validation, label, failures);
        digest.add(copied.data(), copied.size());
        offset += count;
    }
}

template <typename T>
std::vector<T> copy_tensor(const ninfer::Tensor& tensor) {
    std::vector<T> result(static_cast<std::size_t>(tensor.numel()));
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, result.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return result;
}

std::int64_t signed_u64(std::uint64_t value) { return std::bit_cast<std::int64_t>(value); }

std::array<std::int32_t, 16> expected_ple_rows(
    std::int32_t current, const std::array<std::int32_t, 2>& history) {
    const std::int32_t lag1 = history[1];
    const std::int32_t lag2 = lag1 == kEos ? kEos : history[0];
    const std::uint64_t mixed2 = static_cast<std::uint64_t>(current) * kNgramMultiplier[0] ^
                                 static_cast<std::uint64_t>(lag1) * kNgramMultiplier[1];
    const std::uint64_t mixed3 =
        mixed2 ^ static_cast<std::uint64_t>(lag2) * kNgramMultiplier[2];
    std::array<std::int32_t, 16> rows{};
    for (std::size_t head = 0; head < rows.size(); ++head) {
        const std::int64_t mixed = signed_u64(head < 8 ? mixed2 : mixed3);
        std::int64_t remainder = mixed % kNgramPrime[head];
        if (remainder < 0) { remainder += kNgramPrime[head]; }
        rows[head] = kNgramOffset[head] + static_cast<std::int32_t>(remainder);
    }
    return rows;
}

std::int32_t sequence_token(std::int32_t position) {
    const std::uint64_t mixed = 48ULL + static_cast<std::uint64_t>(position) * 104729ULL +
                                static_cast<std::uint64_t>(position) * position * 8191ULL;
    return static_cast<std::int32_t>(mixed % static_cast<std::uint64_t>(kEos));
}

bool is_probe(std::int32_t position) {
    return std::binary_search(kProbePositions.begin(), kProbePositions.end(), position);
}

double decode_e2m1(std::uint8_t nibble) {
    constexpr std::array<double, 8> magnitude = {0.0, 0.5, 1.0, 1.5,
                                                  2.0, 3.0, 4.0, 6.0};
    return (nibble & 8U) == 0 ? magnitude[nibble & 7U] : -magnitude[nibble & 7U];
}

void append_bytes(std::vector<std::uint8_t>& destination, const void* source, std::size_t count) {
    const auto* begin = static_cast<const std::uint8_t*>(source);
    destination.insert(destination.end(), begin, begin + count);
}

template <typename T>
void append_scalar(std::vector<std::uint8_t>& destination, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    append_bytes(destination, &value, sizeof(value));
}

std::vector<std::uint8_t> inspect_qsa_row(const ops::QsaStateView& state, std::int32_t token,
                                          std::size_t layer, int& failures) {
    std::vector<std::uint8_t> image;
    image.reserve(3 * sizeof(std::int32_t) + 128 * sizeof(std::uint16_t) +
                  2 * ops::kQsaKvHeads * (128 + 16));
    std::array<std::int32_t, 3> position{};
    std::array<std::uint16_t, 128> raw_key{};
    const std::size_t position_offset =
        static_cast<std::size_t>(token) * position.size() * sizeof(std::int32_t);
    const std::size_t raw_offset =
        static_cast<std::size_t>(token) * raw_key.size() * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpy(position.data(),
                          static_cast<const std::byte*>(state.positions.data) + position_offset,
                          sizeof(position), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(raw_key.data(),
                          static_cast<const std::byte*>(state.raw_index_keys.data) + raw_offset,
                          sizeof(raw_key), cudaMemcpyDeviceToHost));
    append_bytes(image, position.data(), sizeof(position));
    append_bytes(image, raw_key.data(), sizeof(raw_key));
    if (position != std::array<std::int32_t, 3>{token, token, token}) {
        std::cerr << "QSA position mismatch at token " << token << " layer " << layer << '\n';
        ++failures;
    }
    if (!std::all_of(raw_key.begin(), raw_key.end(), finite_bf16)) {
        std::cerr << "non-finite QSA raw key at token " << token << " layer " << layer << '\n';
        ++failures;
    }

    bool any_nonzero = false;
    for (int head = 0; head < ops::kQsaKvHeads; ++head) {
        std::array<std::uint8_t, 128> k_codes{};
        std::array<std::uint8_t, 128> v_codes{};
        std::array<std::uint8_t, 16> k_scales{};
        std::array<std::uint8_t, 16> v_scales{};
        const std::size_t code_offset = 128ULL *
            (static_cast<std::size_t>(token) + verifier::kQsaCapacity * head);
        const std::size_t scale_offset = 16ULL *
            (static_cast<std::size_t>(token) + verifier::kQsaCapacity * head);
        CUDA_CHECK(cudaMemcpy(k_codes.data(),
                              static_cast<const std::byte*>(state.k_codes.data) + code_offset,
                              k_codes.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_codes.data(),
                              static_cast<const std::byte*>(state.v_codes.data) + code_offset,
                              v_codes.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(k_scales.data(),
                              static_cast<const std::byte*>(state.k_scales.data) + scale_offset,
                              k_scales.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_scales.data(),
                              static_cast<const std::byte*>(state.v_scales.data) + scale_offset,
                              v_scales.size(), cudaMemcpyDeviceToHost));
        append_bytes(image, k_codes.data(), k_codes.size());
        append_bytes(image, v_codes.data(), v_codes.size());
        append_bytes(image, k_scales.data(), k_scales.size());
        append_bytes(image, v_scales.data(), v_scales.size());
        for (int group = 0; group < 16; ++group) {
            __nv_fp8_e4m3 k_scale_bits;
            __nv_fp8_e4m3 v_scale_bits;
            k_scale_bits.__x = k_scales[static_cast<std::size_t>(group)];
            v_scale_bits.__x = v_scales[static_cast<std::size_t>(group)];
            const float k_scale = static_cast<float>(k_scale_bits);
            const float v_scale = static_cast<float>(v_scale_bits);
            if (!std::isfinite(k_scale) || !std::isfinite(v_scale)) {
                std::cerr << "non-finite QSA scale at token " << token << " layer " << layer
                          << '\n';
                ++failures;
            }
            for (int lane = 0; lane < 16; ++lane) {
                const int dimension = 16 * group + lane;
                const std::uint8_t k_packed = k_codes[static_cast<std::size_t>(dimension / 2)];
                const std::uint8_t v_packed = v_codes[static_cast<std::size_t>(dimension / 2)];
                const std::uint8_t k_code =
                    (dimension & 1) == 0 ? k_packed & 15U : k_packed >> 4U;
                const std::uint8_t v_code =
                    (dimension & 1) == 0 ? v_packed & 15U : v_packed >> 4U;
                const double decoded_k = decode_e2m1(k_code) * k_scale;
                const double decoded_v = decode_e2m1(v_code) * v_scale;
                if (!std::isfinite(decoded_k) || !std::isfinite(decoded_v)) {
                    std::cerr << "non-finite decoded QSA row at token " << token << " layer "
                              << layer << '\n';
                    ++failures;
                }
                any_nonzero |= decoded_k != 0.0 || decoded_v != 0.0;
            }
        }
    }
    if (!any_nonzero) {
        std::cerr << "all-zero QSA cache row at token " << token << " layer " << layer << '\n';
        ++failures;
    }
    return image;
}

int validate_selected_ids(std::span<const std::int32_t> ids, std::int32_t count,
                          std::int32_t token, std::size_t layer) {
    const int visible = token + 1;
    const int complete_blocks = visible / 4;
    const int kept_blocks = std::min(complete_blocks, 512);
    const int tail = visible & 3;
    const int expected_count = 4 * kept_blocks + tail;
    int failures = 0;
    if (count != expected_count || ids.size() != static_cast<std::size_t>(ops::kQsaSelectedCapacity)) {
        std::cerr << "QSA selected count/extent mismatch at token " << token << " layer " << layer
                  << " count=" << count << " expected=" << expected_count << '\n';
        return 1;
    }

    std::vector<std::uint8_t> seen(static_cast<std::size_t>(visible), 0);
    std::vector<std::uint8_t> seen_blocks(static_cast<std::size_t>(complete_blocks), 0);
    for (int rank = 0; rank < kept_blocks; ++rank) {
        const std::int32_t first = ids[static_cast<std::size_t>(4 * rank)];
        if (first < 0 || first % 4 != 0 || first + 3 >= 4 * complete_blocks) {
            ++failures;
            continue;
        }
        const int block = first / 4;
        if (seen_blocks[static_cast<std::size_t>(block)] != 0) { ++failures; }
        seen_blocks[static_cast<std::size_t>(block)] = 1;
        for (int lane = 0; lane < 4; ++lane) {
            const std::int32_t id = ids[static_cast<std::size_t>(4 * rank + lane)];
            if (id != first + lane || id < 0 || id >= visible) {
                ++failures;
                continue;
            }
            if (seen[static_cast<std::size_t>(id)] != 0) { ++failures; }
            seen[static_cast<std::size_t>(id)] = 1;
        }
    }
    for (int item = 0; item < tail; ++item) {
        const int destination = 4 * kept_blocks + item;
        const int expected = visible - tail + item;
        if (ids[static_cast<std::size_t>(destination)] != expected ||
            seen[static_cast<std::size_t>(expected)] != 0) {
            ++failures;
        }
        seen[static_cast<std::size_t>(expected)] = 1;
    }
    for (int item = count; item < ops::kQsaSelectedCapacity; ++item) {
        if (ids[static_cast<std::size_t>(item)] != -1) { ++failures; }
    }
    if (complete_blocks <= 512 &&
        !std::all_of(seen.begin(), seen.end(), [](std::uint8_t value) { return value == 1; })) {
        ++failures;
    }
    const bool current_selected = seen[static_cast<std::size_t>(token)] != 0;
    if ((tail != 0 || complete_blocks <= 512) && !current_selected) { ++failures; }
    if (tail != 0 && ids[static_cast<std::size_t>(count - 1)] != token) { ++failures; }
    if (failures != 0) {
        std::cerr << "QSA block/tail/current invariant failed at token " << token << " layer "
                  << layer << " failures=" << failures << '\n';
    }
    return failures;
}

void validate_qsa_layout(const ops::QsaStateView& state, std::size_t layer, int& failures) {
    const auto matches = [](const ninfer::Tensor& tensor, ninfer::DType dtype, int n0, int n1,
                            int n2) {
        return tensor.data != nullptr && tensor.dtype == dtype && tensor.is_contiguous() &&
               tensor.ne[0] == n0 && tensor.ne[1] == n1 && tensor.ne[2] == n2 &&
               tensor.ne[3] == 1;
    };
    if (!matches(state.k_codes, ninfer::DType::U8, 128, verifier::kQsaCapacity, 2) ||
        !matches(state.v_codes, ninfer::DType::U8, 128, verifier::kQsaCapacity, 2) ||
        !matches(state.k_scales, ninfer::DType::FP8_E4M3FN, 16, verifier::kQsaCapacity, 2) ||
        !matches(state.v_scales, ninfer::DType::FP8_E4M3FN, 16, verifier::kQsaCapacity, 2) ||
        !matches(state.raw_index_keys, ninfer::DType::BF16, 128, verifier::kQsaCapacity, 1) ||
        !matches(state.positions, ninfer::DType::I32, 3, verifier::kQsaCapacity, 1)) {
        std::cerr << "malformed QSA state layout at layer " << layer << '\n';
        ++failures;
    }
}

void inspect_probe(const verifier::TokenResultView& result, const verifier::State& state,
                   Digest128& digest, std::vector<std::uint8_t>& exact_transcript,
                   std::array<std::vector<std::uint8_t>, verifier::kQsaLayerCount>& row_zero,
                   bool& row_zero_initialized, int& failures) {
    digest.add_scalar(result.token_index);
    append_scalar(exact_transcript, result.token_index);
    if (result.qsa.size() != verifier::kQsaLayerCount) {
        std::cerr << "QSA diagnostic extent mismatch at token " << result.token_index << '\n';
        ++failures;
        return;
    }
    std::size_t qsa_index = 0;
    for (const verifier::QsaDiagnosticView& diagnostic : result.qsa) {
        const std::size_t expected_layer = 3 + 4 * qsa_index;
        if (diagnostic.layer != expected_layer || !state.qsa()[diagnostic.layer]) {
            std::cerr << "QSA diagnostic layer mismatch at token " << result.token_index << '\n';
            ++failures;
            ++qsa_index;
            continue;
        }
        const auto ids = copy_tensor<std::int32_t>(diagnostic.selected_ids);
        const auto count = copy_tensor<std::int32_t>(diagnostic.selected_count);
        failures += validate_selected_ids(ids, count[0], result.token_index, diagnostic.layer);
        digest.add_scalar(diagnostic.layer);
        digest.add_field("selected_count", count.data(), count.size() * sizeof(count[0]));
        digest.add_field("selected_ids", ids.data(), ids.size() * sizeof(ids[0]));
        append_scalar(exact_transcript, diagnostic.layer);
        append_bytes(exact_transcript, count.data(), count.size() * sizeof(count[0]));
        append_bytes(exact_transcript, ids.data(), ids.size() * sizeof(ids[0]));

        const ops::QsaStateView& qsa_state = *state.qsa()[diagnostic.layer];
        validate_qsa_layout(qsa_state, diagnostic.layer, failures);
        const std::vector<std::uint8_t> current =
            inspect_qsa_row(qsa_state, result.token_index, diagnostic.layer, failures);
        const std::vector<std::uint8_t> first =
            inspect_qsa_row(qsa_state, 0, diagnostic.layer, failures);
        digest.add_field("current_qsa_row", current.data(), current.size());
        digest.add_field("first_qsa_row", first.data(), first.size());
        append_bytes(exact_transcript, current.data(), current.size());
        append_bytes(exact_transcript, first.data(), first.size());
        if (!row_zero_initialized) {
            row_zero[qsa_index] = first;
        } else if (row_zero[qsa_index] != first) {
            std::cerr << "QSA row zero changed by later append at layer " << diagnostic.layer
                      << '\n';
            ++failures;
        }
        ++qsa_index;
    }
    row_zero_initialized = true;
}

Digest128 hash_continuation(const verifier::State& state,
                            const std::array<std::int32_t, 2>& expected_history,
                            std::vector<std::uint8_t>& scratch, int& failures) {
    Digest128 digest;
    std::size_t bytes_hashed = 0;
    std::size_t gdn_count = 0;
    std::size_t qsa_count = 0;
    for (std::size_t layer = 0; layer < verifier::kLayerCount; ++layer) {
        digest.add_scalar(layer);
        if (state.gdn()[layer]) {
            ++gdn_count;
            hash_device_tensor(digest, state.gdn()[layer]->conv, "gdn_conv",
                               ValidationKind::Bf16, scratch, failures, &bytes_hashed);
            hash_device_tensor(digest, state.gdn()[layer]->recurrence, "gdn_recurrence",
                               ValidationKind::Fp32, scratch, failures, &bytes_hashed);
        }
        if (state.qsa()[layer]) {
            ++qsa_count;
            const ops::QsaStateView& qsa = *state.qsa()[layer];
            validate_qsa_layout(qsa, layer, failures);
            hash_device_tensor(digest, qsa.k_codes, "qsa_k_codes", ValidationKind::None,
                               scratch, failures, &bytes_hashed);
            hash_device_tensor(digest, qsa.v_codes, "qsa_v_codes", ValidationKind::None,
                               scratch, failures, &bytes_hashed);
            hash_device_tensor(digest, qsa.k_scales, "qsa_k_scales",
                               ValidationKind::Fp8E4m3, scratch, failures, &bytes_hashed);
            hash_device_tensor(digest, qsa.v_scales, "qsa_v_scales",
                               ValidationKind::Fp8E4m3, scratch, failures, &bytes_hashed);
            hash_device_tensor(digest, qsa.raw_index_keys, "qsa_raw_keys",
                               ValidationKind::Bf16, scratch, failures, &bytes_hashed);
            const auto positions = copy_tensor<std::int32_t>(qsa.positions);
            digest.add_field("qsa_positions", positions.data(),
                             positions.size() * sizeof(positions[0]));
            bytes_hashed += qsa.positions.bytes();
            for (std::int32_t token = 0; token < verifier::kQsaCapacity; ++token) {
                for (int axis = 0; axis < 3; ++axis) {
                    if (positions[static_cast<std::size_t>(3 * token + axis)] != token) {
                        std::cerr << "QSA final position plane mismatch at token " << token
                                  << " layer " << layer << '\n';
                        ++failures;
                        token = verifier::kQsaCapacity;
                        break;
                    }
                }
            }
        }
    }
    hash_device_tensor(digest, state.ple_conv_state(), "ple_conv", ValidationKind::Bf16,
                       scratch, failures, &bytes_hashed);
    const auto history = copy_tensor<std::int32_t>(state.ple_token_history());
    digest.add_field("ple_history", history.data(), history.size() * sizeof(history[0]));
    bytes_hashed += state.ple_token_history().bytes();
    if (!std::equal(history.begin(), history.end(), expected_history.begin())) {
        std::cerr << "final PLE history mismatch\n";
        ++failures;
    }
    hash_device_tensor(digest, state.residual(), "residual", ValidationKind::Bf16, scratch,
                       failures, &bytes_hashed);
    const std::size_t expected_bytes = verifier::kPersistentStateBytes + state.residual().bytes();
    if (gdn_count != verifier::kGdnLayerCount || qsa_count != verifier::kQsaLayerCount ||
        bytes_hashed != expected_bytes) {
        std::cerr << "continuation inventory mismatch: gdn=" << gdn_count << " qsa=" << qsa_count
                  << " bytes=" << bytes_hashed << " expected=" << expected_bytes << '\n';
        ++failures;
    }
    return digest;
}

template <typename Visitor>
void visit_continuation_tensors(const verifier::State& state, Visitor&& visitor) {
    for (std::size_t layer = 0; layer < verifier::kLayerCount; ++layer) {
        if (state.gdn()[layer]) {
            visitor("gdn_conv", state.gdn()[layer]->conv);
            visitor("gdn_recurrence", state.gdn()[layer]->recurrence);
        }
        if (state.qsa()[layer]) {
            const ops::QsaStateView& qsa = *state.qsa()[layer];
            visitor("qsa_k_codes", qsa.k_codes);
            visitor("qsa_v_codes", qsa.v_codes);
            visitor("qsa_k_scales", qsa.k_scales);
            visitor("qsa_v_scales", qsa.v_scales);
            visitor("qsa_raw_keys", qsa.raw_index_keys);
            visitor("qsa_positions", qsa.positions);
        }
    }
    visitor("ple_conv", state.ple_conv_state());
    visitor("ple_history", state.ple_token_history());
    visitor("residual", state.residual());
}

std::vector<std::uint8_t> capture_continuation(const verifier::State& state) {
    const std::size_t expected_bytes = verifier::kPersistentStateBytes + state.residual().bytes();
    std::vector<std::uint8_t> image;
    image.reserve(expected_bytes);
    visit_continuation_tensors(state, [&](std::string_view, const ninfer::Tensor& tensor) {
        const std::size_t offset = image.size();
        image.resize(offset + tensor.bytes());
        CUDA_CHECK(cudaMemcpy(image.data() + offset, tensor.data, tensor.bytes(),
                              cudaMemcpyDeviceToHost));
    });
    if (image.size() != expected_bytes) {
        throw std::runtime_error("captured continuation inventory has an unexpected byte count");
    }
    return image;
}

bool continuation_equals(const verifier::State& state,
                         std::span<const std::uint8_t> expected,
                         std::vector<std::uint8_t>& scratch) {
    const std::size_t expected_bytes = verifier::kPersistentStateBytes + state.residual().bytes();
    if (expected.size() != expected_bytes) {
        std::cerr << "continuation reference byte count mismatch: actual=" << expected.size()
                  << " expected=" << expected_bytes << '\n';
        return false;
    }
    bool equal = true;
    std::size_t image_offset = 0;
    visit_continuation_tensors(
        state, [&](std::string_view label, const ninfer::Tensor& tensor) {
            for (std::size_t tensor_offset = 0; tensor_offset < tensor.bytes();) {
                const std::size_t count =
                    std::min(scratch.size(), tensor.bytes() - tensor_offset);
                CUDA_CHECK(cudaMemcpy(scratch.data(),
                                      static_cast<const std::byte*>(tensor.data) + tensor_offset,
                                      count, cudaMemcpyDeviceToHost));
                if (std::memcmp(scratch.data(), expected.data() + image_offset + tensor_offset,
                                count) != 0) {
                    if (equal) {
                        std::cerr << "exact continuation mismatch in " << label
                                  << " at global byte " << image_offset + tensor_offset << '\n';
                    }
                    equal = false;
                }
                tensor_offset += count;
            }
            image_offset += tensor.bytes();
        });
    if (image_offset != expected.size()) {
        std::cerr << "continuation traversal byte count mismatch: actual=" << image_offset
                  << " expected=" << expected.size() << '\n';
        return false;
    }
    return equal;
}

struct RunEvidence {
    Digest128 observables;
    Digest128 probes;
    std::vector<std::uint8_t> exact_probes;
    Digest128 continuation;

    bool operator==(const RunEvidence&) const = default;
};

RunEvidence run_sequence(verifier::Program& program, int replay,
                         std::vector<std::uint8_t>* captured_continuation,
                         const std::vector<std::uint8_t>* expected_continuation,
                         int& failures) {
    program.reset();
    if (program.frontier() != 0) {
        std::cerr << "Program reset did not restore frontier zero\n";
        ++failures;
    }
    std::vector<std::uint8_t> scratch(kCopyChunkBytes);
    Digest128 observables;
    Digest128 probes;
    std::vector<std::uint8_t> exact_probes;
    exact_probes.reserve(kProbePositions.size() * verifier::kQsaLayerCount * 10000U);
    std::array<std::vector<std::uint8_t>, verifier::kQsaLayerCount> row_zero{};
    bool row_zero_initialized = false;
    std::array<std::int32_t, 2> ple_history = {kEos, kEos};
    const auto begin = std::chrono::steady_clock::now();

    for (std::int32_t position = 0; position < verifier::kQsaCapacity; ++position) {
        const std::int32_t token = sequence_token(position);
        const std::int32_t target = sequence_token(position + 1);
        const auto expected_rows = expected_ple_rows(token, ple_history);
        ple_history = {ple_history[1], token};
        const verifier::TokenResultView result = program.execute_token(token, target);
        if (result.token_index != position || program.frontier() != position + 1 ||
            !result.gr.empty() || result.qsa.size() != verifier::kQsaLayerCount ||
            result.routers.size() != verifier::kLayerCount) {
            std::cerr << "Program extent/frontier mismatch at token " << position << '\n';
            ++failures;
        }

        observables.add_scalar(position);
        observables.add_scalar(token);
        observables.add_scalar(target);
        hash_device_tensor(observables, result.logits, "logits", ValidationKind::Bf16, scratch,
                           failures);
        hash_device_tensor(observables, result.final_hidden, "final_hidden",
                           ValidationKind::Bf16, scratch, failures);
        const auto nll = copy_tensor<float>(result.nll);
        observables.add_field("nll", nll.data(), nll.size() * sizeof(nll[0]));
        if (nll.size() != 1 || !std::isfinite(nll[0]) || nll[0] < 0.0F) {
            std::cerr << "invalid NLL at token " << position << '\n';
            ++failures;
        }
        const auto ple_rows = copy_tensor<std::int32_t>(result.ple_row_ids);
        observables.add_field("ple_rows", ple_rows.data(), ple_rows.size() * sizeof(ple_rows[0]));
        if (ple_rows.size() != expected_rows.size() ||
            !std::equal(ple_rows.begin(), ple_rows.end(), expected_rows.begin())) {
            std::cerr << "PLE row mismatch at token " << position << '\n';
            ++failures;
        }
        const auto represented_history = copy_tensor<std::int32_t>(program.state().ple_token_history());
        observables.add_field("ple_history", represented_history.data(),
                              represented_history.size() * sizeof(represented_history[0]));
        if (!std::equal(represented_history.begin(), represented_history.end(),
                        ple_history.begin())) {
            std::cerr << "PLE history mismatch at token " << position << '\n';
            ++failures;
        }

        if (is_probe(position)) {
            inspect_probe(result, program.state(), probes, exact_probes, row_zero,
                          row_zero_initialized, failures);
            std::cout << "Qwen4 long replay=" << replay << " probe=" << position
                      << " visible=" << position + 1 << '\n';
        }
    }

    const Digest128 continuation =
        hash_continuation(program.state(), ple_history, scratch, failures);
    if (captured_continuation != nullptr) {
        *captured_continuation = capture_continuation(program.state());
    }
    if (expected_continuation != nullptr &&
        !continuation_equals(program.state(), *expected_continuation, scratch)) {
        std::cerr << "final continuation differs byte-for-byte after reset/replay\n";
        ++failures;
    }
    if (program.frontier() != verifier::kQsaCapacity) {
        std::cerr << "Program did not reach exact capacity frontier\n";
        ++failures;
    }
    bool rejected = false;
    try {
        (void)program.execute_token(sequence_token(verifier::kQsaCapacity),
                                    sequence_token(verifier::kQsaCapacity + 1));
    } catch (const std::length_error&) {
        rejected = true;
    }
    if (!rejected || program.frontier() != verifier::kQsaCapacity) {
        std::cerr << "Program did not reject token beyond capacity before frontier mutation\n";
        ++failures;
    }
    const Digest128 after_rejection = hash_continuation(program.state(), ple_history, scratch,
                                                         failures);
    if (!(continuation == after_rejection)) {
        std::cerr << "Program continuation hash changed after rejected capacity overflow\n";
        ++failures;
    }
    const std::vector<std::uint8_t>* rejection_reference =
        captured_continuation != nullptr ? captured_continuation : expected_continuation;
    if (rejection_reference == nullptr ||
        !continuation_equals(program.state(), *rejection_reference, scratch)) {
        std::cerr << "Program state changed byte-for-byte after rejected capacity overflow\n";
        ++failures;
    }

    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
                               .count();
    std::cout << "Qwen4 long replay=" << replay << " seconds=" << seconds
              << " observable_hash=" << digest_string(observables)
              << " probe_hash=" << digest_string(probes)
              << " continuation_hash=" << digest_string(continuation)
              << " exact_probe_bytes=" << exact_probes.size()
              << " exact_continuation_bytes=" << rejection_reference->size() << '\n';
    return {observables, probes, std::move(exact_probes), continuation};
}

} // namespace

int main() {
    const char* configured = std::getenv("NINFER_QWEN4_VERIFY_WEIGHTS");
    if (configured == nullptr || *configured == '\0') {
        std::cout << "skip: NINFER_QWEN4_VERIFY_WEIGHTS is not set\n";
        return 77;
    }
    const std::filesystem::path path(configured);
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "verifier artifact is not a regular file: " << path << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        std::unique_ptr<verifier::LoadedModel> model = verifier::LoadedModel::load(path, device);
        verifier::Program program(*model, device, verifier::DiagnosticSnapshots::Disabled);
        int failures = 0;
        std::vector<std::uint8_t> first_continuation;
        const RunEvidence first = run_sequence(program, 0, &first_continuation, nullptr, failures);
        const RunEvidence replay =
            run_sequence(program, 1, nullptr, &first_continuation, failures);
        if (!(first == replay)) {
            std::cerr << "long-frontier observable/probe reset-replay mismatch\n";
            ++failures;
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " qwen4_program_long_real tokens_per_replay=" << verifier::kQsaCapacity
                  << " replays=2 probes=" << kProbePositions.size() << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Qwen4 long-frontier real-artifact execution failed: " << error.what()
                  << '\n';
        return 1;
    }
}
