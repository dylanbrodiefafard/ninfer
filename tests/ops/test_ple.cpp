#include "ninfer/ops/ple.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/native_projection_fixture.h"
#include "ops/ple_nvfp4_oracle.h"
#include "artifact/typed_binding.h"
#include "artifact/materializer.h"
#include "targets/qwen4/native_bf16_fixture.h"
#include "targets/qwen4/native_sequence_components.h"
#include "targets/qwen4/native_text_panel.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <sys/mman.h>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::size_t kQ8RowBytes = (ops::kPleEmbeddingWidth / 32) * 34;

std::uint16_t fp8_ple_oracle(unsigned code, std::uint16_t scale) {
    const int exponent=(code>>3)&15, mantissa=code&7;
    const double magnitude=exponent==0 ? std::ldexp(double(mantissa),-9)
        : std::ldexp(1+double(mantissa)/8,exponent-7);
    const double value=std::copysign(magnitude,(code&128)?-1.0:1.0);
    return f32_to_bf16(float(value*bf16_to_f32(scale)));
}

int nvfp4_staging_decode_case(int width) {
    constexpr std::size_t partitions = 5, rows_per_partition = 17;
    constexpr std::size_t rows = partitions * rows_per_partition;
    constexpr std::size_t table_bytes = rows * 90 + partitions * 4;
    auto* table = static_cast<std::uint8_t*>(mmap(nullptr, table_bytes,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (table == MAP_FAILED) { throw std::runtime_error("NVFP4 PLE mmap failed"); }
    struct Mapping { void* data; ~Mapping() { munlock(data, table_bytes); munmap(data, table_bytes); } } mapping{table};
    if (mlock(table, table_bytes)) { throw std::runtime_error("NVFP4 PLE mlock failed"); }
    // Unequal source-like scales, exact BF16 midpoint, F32/BF16 subnormals and large finite value.
    constexpr std::uint32_t multipliers[]{0x37b30c31, 0x379f3cf3, 0x3f808000, 0x00400000, 0x70000000};
    for (std::size_t p = 0; p < partitions; ++p) {
        for (int b = 0; b < 4; ++b) { table[rows * 90 + p * 4 + b] = multipliers[p] >> (8 * b); }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (int pair = 0; pair < 80; ++pair) {
            table[row * 90 + pair] = ((pair + row) % 16) | (((3 * pair + row) % 16) << 4);
        }
        for (int group = 0; group < 10; ++group) { table[row * 90 + 80 + group] = (row * 10 + group) % 127; }
    }
    std::vector<std::int32_t> ids(static_cast<std::size_t>(width) * 16);
    for (std::size_t i = 0; i < ids.size(); ++i) { ids[i] = (i * 19 + i / 16) % rows; }
    ids[0] = 0; ids[1] = rows_per_partition - 1; ids[2] = rows_per_partition;
    ids[3] = rows - 1; ids[4] = ids[3];
    const std::size_t packed_bytes = ids.size() * 94, count = ids.size() * 160;
    PinnedHostBuffer pinned(packed_bytes);
    GuardedDeviceBuffer packed(packed_bytes), decoded(count * 2);
    Tensor encoded(packed.data(), DType::U8, {94, 16, width});
    Tensor output(decoded.data(), DType::BF16, {160, 16, width});
    const ops::PleResidentNvfp4Table view{table, partitions, rows_per_partition, table_bytes};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    struct Stream { cudaStream_t value; ~Stream() { cudaStreamDestroy(value); } } owner{stream};
    std::vector<std::uint8_t> expected_bytes(packed_bytes);
    std::vector<std::uint16_t> expected(count);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        auto* record = expected_bytes.data() + i * 94;
        for (int b = 0; b < 90; ++b) { record[b] = table[ids[i] * 90 + b]; }
        const auto word = multipliers[ids[i] / rows_per_partition];
        for (int b = 0; b < 4; ++b) { record[90 + b] = word >> (8 * b); }
        for (int f = 0; f < 160; ++f) { expected[i * 160 + f] = ple_nvfp4_oracle(record, f); }
    }
    ops::ple_nvfp4_stage_rows_batch(view, ids, width, pinned.data(), pinned.size(), encoded, stream);
    ops::ple_nvfp4_decode_rows(encoded, output, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int failures = verify_exact("NVFP4 PLE packed rows and partition scales",
        from_device<std::uint8_t>(packed.data(), packed_bytes), expected_bytes);
    failures += verify_exact("NVFP4 PLE independent exact codec",
        from_device<std::uint16_t>(decoded.data(), count), expected);
    if (width == 17) {
        auto invalid = ids;
        invalid.back() = rows;
        bool rejected = false;
        try { ops::ple_nvfp4_stage_rows_batch(view, invalid, width, pinned.data(), pinned.size(), encoded, stream); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) { ++failures; }
        std::vector<std::uint8_t> actual_pinned(packed_bytes);
        std::memcpy(actual_pinned.data(), pinned.data(), packed_bytes);
        failures += verify_exact("NVFP4 PLE invalid id preserves pinned slot", actual_pinned, expected_bytes);
        for (int start = 0; start < width; start += 3) {
            const int n = std::min(3, width - start);
            Tensor chunk(packed.data(), DType::U8, {94, 16, n});
            Tensor result(static_cast<std::uint16_t*>(decoded.data()) + start * 2560,
                DType::BF16, {160, 16, n});
            ops::ple_nvfp4_stage_rows_batch(view, std::span(ids).subspan(start * 16, n * 16),
                n, pinned.data(), pinned.size(), chunk, stream);
            ops::ple_nvfp4_decode_rows(chunk, result, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        failures += verify_exact("NVFP4 PLE bounded slot reuse and chunks",
            from_device<std::uint16_t>(decoded.data(), count), expected);
    }
    failures += packed.verify_guards("NVFP4 PLE staged records");
    failures += decoded.verify_guards("NVFP4 PLE BF16 output");
    return failures;
}

int fp8_staging_decode_case(int width) {
    constexpr std::size_t rows = 19;
    constexpr std::size_t table_bytes = rows * ops::kPleRowWidth;
    auto* table = static_cast<std::uint8_t*>(mmap(nullptr, table_bytes,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (table == MAP_FAILED) { throw std::runtime_error("FP8 PLE test mmap failed"); }
    struct Mapping {
        void* data;
        ~Mapping() { munlock(data, table_bytes); munmap(data, table_bytes); }
    } mapping{table};
    if (mlock(table, table_bytes) != 0) { throw std::runtime_error("FP8 PLE test mlock failed"); }
    for (std::size_t i = 0; i < table_bytes; ++i) {
        // All 254 finite E4M3FN codes, including both zeros and the extended exponent-15 range.
        const auto ordinal = i % 254;
        table[i] = static_cast<std::uint8_t>(ordinal < 127 ? ordinal : ordinal + 1);
    }
    std::vector<std::int32_t> ids(static_cast<std::size_t>(width) * ops::kPleHeads);
    for (std::size_t i = 0; i < ids.size(); ++i) { ids[i] = (i * 7 + i / 16) % rows; }
    const std::size_t count = ids.size() * ops::kPleRowWidth;
    PinnedHostBuffer pinned(count);
    GuardedDeviceBuffer staged(count), output(count * 2);
    Tensor encoded(staged.data(), DType::U8, {ops::kPleRowWidth, ops::kPleHeads, width});
    Tensor decoded(output.data(), DType::BF16, {ops::kPleRowWidth, ops::kPleHeads, width});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    struct Stream { cudaStream_t value; ~Stream() { cudaStreamDestroy(value); } } owner{stream};
    const ops::PleResidentFp8Table view{table, rows, table_bytes};
    ops::ple_fp8_stage_rows_batch(view, ids, width, pinned.data(), pinned.size(), encoded, stream);
    std::vector<std::uint8_t> expected_bytes(count);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::memcpy(expected_bytes.data() + i * ops::kPleRowWidth,
                    table + ids[i] * ops::kPleRowWidth, ops::kPleRowWidth);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int failures = verify_exact("FP8 PLE packed row order", from_device<std::uint8_t>(staged.data(), count), expected_bytes);
    if (width == 17) {
        auto invalid = ids;
        invalid.back() = rows;
        bool rejected = false;
        try {
            ops::ple_fp8_stage_rows_batch(view, invalid, width, pinned.data(), pinned.size(), encoded, stream);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) { ++failures; }
        std::vector<std::uint8_t> pinned_after(count);
        std::memcpy(pinned_after.data(), pinned.data(), count);
        failures += verify_exact("FP8 PLE invalid row preserves pinned slot", pinned_after, expected_bytes);
        for (const std::uint16_t invalid_scale : {0, 0x8000, 0x7f80, 0x7fc0}) {
            rejected = false;
            try { ops::ple_fp8_decode_rows(encoded, invalid_scale, decoded, stream); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) { ++failures; }
        }
    }
    for (const std::uint16_t scale : {0x3951, 0x3f80, 0x4001, 0x0080, 0x0001}) {
        ops::ple_fp8_decode_rows(encoded, scale, decoded, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> expected(count);
        for (std::size_t i = 0; i < count; ++i) {
            expected[i] = fp8_ple_oracle(expected_bytes[i], scale);
        }
        failures += verify_exact("FP8 PLE exact scaled BF16 decode",
            from_device<std::uint16_t>(output.data(), count), expected);
        // Independent chunked staging/decoding into the same output, preserving token order.
        for (int start = 0; width == 17 && start < width; start += 3) {
            const int n = std::min(3, width - start);
            Tensor chunk(staged.data(), DType::U8, {ops::kPleRowWidth, ops::kPleHeads, n});
            Tensor result(static_cast<std::uint16_t*>(output.data()) + start * ops::kPleEmbeddingWidth,
                DType::BF16, {ops::kPleRowWidth, ops::kPleHeads, n});
            ops::ple_fp8_stage_rows_batch(view, std::span(ids).subspan(start * 16, n * 16),
                n, pinned.data(), pinned.size(), chunk, stream);
            ops::ple_fp8_decode_rows(chunk, scale, result, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream)); // slot reuse only after all consumers drain
        }
        failures += verify_exact("FP8 PLE chunk boundaries",
            from_device<std::uint16_t>(output.data(), count), expected);
        // Restore packed full panel before the next scale.
        ops::ple_fp8_stage_rows_batch(view, ids, width, pinned.data(), pinned.size(), encoded, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    failures += staged.verify_guards("FP8 PLE packed rows");
    failures += output.verify_guards("FP8 PLE decoded rows");
    return failures;
}

std::uint16_t f16_bits(float value) {
    const __half encoded = __float2half_rn(value);
    std::uint16_t bits;
    std::memcpy(&bits, &encoded, sizeof(bits));
    return bits;
}

float f16_value(std::uint16_t bits) {
    __half encoded;
    std::memcpy(&encoded, &bits, sizeof(bits));
    return __half2float(encoded);
}

ops::PleMappedIq4NlTable mapped_view(const std::vector<std::uint8_t>& table, std::uint64_t rows) {
    return {table.data(), rows, table.size()};
}

double iq4_oracle(const std::uint8_t* block, int index) {
    constexpr int codebook[]{-127, -104, -83, -65, -49, -35, -22, -10,
                             1,    13,   25,  38,  53,  69,  89,  113};
    const std::uint16_t scale_bits =
        static_cast<std::uint16_t>(block[0]) | (static_cast<std::uint16_t>(block[1]) << 8U);
    const std::uint8_t packed = block[2 + (index & 15)];
    const int code = index < 16 ? packed & 0x0fU : packed >> 4U;
    return static_cast<double>(f16_value(scale_bits)) * codebook[code];
}


int staging_decode_case() {
    constexpr int rows = 23;
    std::vector<std::uint8_t> table(static_cast<std::size_t>(rows) * ops::kPleIq4NlRowBytes);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < 5; ++block) {
            auto* encoded = table.data() + static_cast<std::size_t>(row) * ops::kPleIq4NlRowBytes +
                            block * ops::kPleIq4NlBlockBytes;
            const std::uint16_t scale = f16_bits((row + block + 1) / 256.0F);
            encoded[0] = static_cast<std::uint8_t>(scale);
            encoded[1] = static_cast<std::uint8_t>(scale >> 8U);
            for (int lane = 0; lane < 16; ++lane) {
                encoded[2 + lane] = static_cast<std::uint8_t>(((row + block + lane + 7) & 15) << 4U) |
                                     static_cast<std::uint8_t>((row + 3 * block + lane) & 15);
            }
        }
    }
    const std::array<std::int32_t, ops::kPleHeads> ids{
        22, 0, 17, 3, 3, 9, 1, 21, 4, 16, 8, 12, 6, 19, 2, 14,
    };
    PinnedHostBuffer pinned(ops::kPleStagedBytes);
    GuardedDeviceBuffer staged(ops::kPleStagedBytes);
    GuardedDeviceBuffer output(static_cast<std::size_t>(ops::kPleHeads) * ops::kPleRowWidth *
                               sizeof(std::uint16_t));
    Tensor staged_tensor(staged.data(), DType::U8, {ops::kPleIq4NlRowBytes, ops::kPleHeads});
    Tensor output_tensor(output.data(), DType::BF16, {ops::kPleRowWidth, ops::kPleHeads});
    ops::ple_iq4_nl_stage_rows(mapped_view(table, rows), ids, pinned.data(), pinned.size(),
                               staged_tensor, nullptr);
    ops::ple_iq4_nl_decode_rows(staged_tensor, output_tensor, nullptr);
    cuda_synchronize();

    std::vector<std::uint8_t> expected_staged(ops::kPleStagedBytes);
    for (int head = 0; head < ops::kPleHeads; ++head) {
        std::memcpy(expected_staged.data() + head * ops::kPleIq4NlRowBytes,
                    table.data() + static_cast<std::size_t>(ids[head]) * ops::kPleIq4NlRowBytes,
                    ops::kPleIq4NlRowBytes);
    }
    std::vector<std::uint8_t> actual_pinned(ops::kPleStagedBytes);
    std::memcpy(actual_pinned.data(), pinned.data(), actual_pinned.size());
    int failures = verify_exact("PLE pinned selected rows", actual_pinned, expected_staged);
    failures += verify_exact("PLE device selected rows",
                             from_device<std::uint8_t>(staged.data(), ops::kPleStagedBytes),
                             expected_staged);

    const auto actual = from_device<std::uint16_t>(output.data(),
                                                   static_cast<std::size_t>(ops::kPleHeads) *
                                                       ops::kPleRowWidth);
    std::vector<std::uint16_t> expected(actual.size());
    for (int head = 0; head < ops::kPleHeads; ++head) {
        const auto* row = table.data() + static_cast<std::size_t>(ids[head]) * ops::kPleIq4NlRowBytes;
        for (int d = 0; d < ops::kPleRowWidth; ++d) {
            const auto* block = row + (d / 32) * 18;
            expected[d + ops::kPleRowWidth * head] =
                f32_to_bf16(static_cast<float>(iq4_oracle(block, d % 32)));
        }
    }
    failures += verify_exact("PLE exact IQ4_NL decode", actual, expected);
    failures += staged.verify_guards("PLE staged rows");
    failures += output.verify_guards("PLE decoded embedding");
    return failures;
}

int batched_staging_decode_case(int width) {
    constexpr int rows = 41;
    const std::size_t slots = static_cast<std::size_t>(ops::kPleHeads) * width;
    const std::size_t staged_bytes = static_cast<std::size_t>(ops::kPleStagedBytes) * width;
    std::vector<std::uint8_t> table(static_cast<std::size_t>(rows) * ops::kPleIq4NlRowBytes);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < 5; ++block) {
            auto* encoded = table.data() + static_cast<std::size_t>(row) * ops::kPleIq4NlRowBytes +
                            block * ops::kPleIq4NlBlockBytes;
            const std::uint16_t scale = f16_bits((2 * row + block + 3) / 512.0F);
            encoded[0] = static_cast<std::uint8_t>(scale);
            encoded[1] = static_cast<std::uint8_t>(scale >> 8U);
            for (int lane = 0; lane < 16; ++lane) {
                const int low = (row + 5 * block + lane) & 15;
                const int high = (3 * row + block + 2 * lane + 1) & 15;
                encoded[2 + lane] = static_cast<std::uint8_t>(low | (high << 4));
            }
        }
    }
    std::vector<std::int32_t> ids(slots);
    for (int token = 0; token < width; ++token) {
        for (int head = 0; head < ops::kPleHeads; ++head) {
            ids[head + ops::kPleHeads * token] =
                (token == width - 1 && (head == 3 || head == 11))
                    ? 7
                    : (17 * token + 5 * head + 2) % rows;
        }
    }

    PinnedHostBuffer pinned(staged_bytes);
    GuardedDeviceBuffer staged(staged_bytes);
    GuardedDeviceBuffer output(slots * ops::kPleRowWidth * sizeof(std::uint16_t));
    Tensor staged_tensor(staged.data(), DType::U8,
                         {ops::kPleIq4NlRowBytes, ops::kPleHeads, width});
    Tensor output_tensor(output.data(), DType::BF16,
                         {ops::kPleRowWidth, ops::kPleHeads, width});
    ops::ple_iq4_nl_stage_rows_batch(mapped_view(table, rows), ids, width, pinned.data(),
                                     pinned.size(), staged_tensor, nullptr);
    ops::ple_iq4_nl_decode_rows(staged_tensor, output_tensor, nullptr);
    cuda_synchronize();

    std::vector<std::uint8_t> expected_staged(staged_bytes);
    std::vector<std::uint16_t> expected(slots * ops::kPleRowWidth);
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto* row = table.data() + static_cast<std::size_t>(ids[slot]) * ops::kPleIq4NlRowBytes;
        std::memcpy(expected_staged.data() + slot * ops::kPleIq4NlRowBytes, row,
                    ops::kPleIq4NlRowBytes);
        for (int d = 0; d < ops::kPleRowWidth; ++d) {
            expected[d + ops::kPleRowWidth * slot] = f32_to_bf16(static_cast<float>(
                iq4_oracle(row + (d / 32) * ops::kPleIq4NlBlockBytes, d % 32)));
        }
    }
    std::vector<std::uint8_t> actual_pinned(staged_bytes);
    std::memcpy(actual_pinned.data(), pinned.data(), staged_bytes);
    int failures = verify_exact("PLE batched pinned row bytes", actual_pinned, expected_staged);
    failures += verify_exact("PLE batched device row bytes",
                             from_device<std::uint8_t>(staged.data(), staged_bytes),
                             expected_staged);
    failures += verify_exact("PLE batched exact IQ4_NL decode",
                             from_device<std::uint16_t>(output.data(), expected.size()), expected);

    const std::vector<std::uint8_t> before =
        from_device<std::uint8_t>(staged.data(), staged_bytes);
    auto invalid_ids = ids;
    invalid_ids[ops::kPleHeads + 9] = rows;
    bool rejected = false;
    try {
        ops::ple_iq4_nl_stage_rows_batch(mapped_view(table, rows), invalid_ids, width,
                                         pinned.data(), pinned.size(), staged_tensor, nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "FAIL PLE batched invalid row was accepted\n";
        ++failures;
    }
    failures += verify_exact("PLE rejected batch leaves device staging unchanged",
                             from_device<std::uint8_t>(staged.data(), staged_bytes), before);
    rejected = false;
    try {
        ops::ple_iq4_nl_stage_rows_batch(mapped_view(table, rows), ids, width, pinned.data(),
                                         staged_bytes - 1, staged_tensor, nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "FAIL PLE batched short pinned slot was accepted\n";
        ++failures;
    }
    rejected = false;
    try {
        ops::ple_iq4_nl_stage_rows_batch(mapped_view(table, rows),
                                         std::span<const std::int32_t>(ids).first(ids.size() - 1),
                                         width, pinned.data(), pinned.size(), staged_tensor,
                                         nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "FAIL PLE batched malformed row-id extent was accepted\n";
        ++failures;
    }
    failures += staged.verify_guards("PLE batched staged rows");
    failures += output.verify_guards("PLE batched decoded embedding");
    return failures;
}

struct Weights {
    Weights(std::vector<std::uint16_t> key_bits, std::vector<std::uint16_t> value_bits) {
        const auto k=qwen4_native::bf16_matrix(key_bits,10240,2560);
        const auto v=qwen4_native::bf16_matrix(value_bits,2560,2560);
        key_device=to_device(k.payload); value_device=to_device(v.payload);
        key=k.device_weight(key_device.p); value=v.device_weight(value_device.p);
        source_key.resize(key_bits.size()); source_value.resize(value_bits.size());
        std::transform(key_bits.begin(),key_bits.end(),source_key.begin(),bf16_to_f32);
        std::transform(value_bits.begin(),value_bits.end(),source_value.begin(),bf16_to_f32);
    }

    std::pair<std::vector<double>,std::vector<double>> project(std::span<const float> input) const {
        std::vector<double> k(10240),v(2560);
        if(source_key.empty()) {
            for(int row=0;row<10240;++row) { k[row]=((row&1)?-1:1)*input[0]*key_unit; }
            std::fill(v.begin(),v.end(),input[1]*value_unit);
        } else {
            for(int row=0;row<10240;++row) for(int d=0;d<2560;++d) {
                k[row]+=double(source_key[row*2560+d])*double(input[d]);
            }
            for(int row=0;row<2560;++row) for(int d=0;d<2560;++d) {
                v[row]+=double(source_value[row*2560+d])*double(input[d]);
            }
        }
        return {std::move(k),std::move(v)};
    }

    Weights(QType type = QType::GGML_Q8_0)
        : key_host(static_cast<std::size_t>(ops::kPleChannels) * kQ8RowBytes, 0),
          value_host(static_cast<std::size_t>(ops::kPleEmbeddingWidth) * kQ8RowBytes, 0),
          key_device(key_host.size()), value_device(value_host.size()) {
        scale_bits = f16_bits(1.0F / 127.0F);
        scale = f16_value(scale_bits);
        for (int row = 0; row < ops::kPleChannels; ++row) {
            auto* block = key_host.data() + static_cast<std::size_t>(row) * kQ8RowBytes;
            block[0] = static_cast<std::uint8_t>(scale_bits);
            block[1] = static_cast<std::uint8_t>(scale_bits >> 8U);
            block[2] = static_cast<std::uint8_t>((row & 1) == 0 ? 127 : 129);
        }
        for (int row = 0; row < ops::kPleEmbeddingWidth; ++row) {
            auto* block = value_host.data() + static_cast<std::size_t>(row) * kQ8RowBytes;
            block[0] = static_cast<std::uint8_t>(scale_bits);
            block[1] = static_cast<std::uint8_t>(scale_bits >> 8U);
            block[3] = 64;
        }
        key_device.copy_from_host(key_host.data(), key_host.size());
        value_device.copy_from_host(value_host.data(), value_host.size());
        key = make_weight(key_device, ops::kPleChannels);
        value = make_weight(value_device, ops::kPleEmbeddingWidth);
        key_unit = static_cast<double>(scale) * 127.0;
        value_unit = static_cast<double>(scale) * 64.0;
        if (native_projection_format(type)) {
            auto native_key = native_sparse_fixture(type, ops::kPleChannels, ops::kPleEmbeddingWidth);
            auto native_value = native_sparse_fixture(type, ops::kPleEmbeddingWidth, ops::kPleEmbeddingWidth);
            for (int row = 0; row < ops::kPleChannels; ++row) {
                native_sparse_set(native_key, row, 0, (row & 1) ? -1.0F : 1.0F);
            }
            for (int row = 0; row < ops::kPleEmbeddingWidth; ++row) {
                native_sparse_set(native_value, row, 1, 0.5F);
            }
            key_device = to_device(native_key.payload);
            value_device = to_device(native_value.payload);
            key = native_key.device_weight(key_device.p);
            value = native_value.device_weight(value_device.p);
            key_unit = 1.0;
            value_unit = 0.5;
        }
    }

    static Weight make_weight(DeviceBuffer& storage, int rows) {
        Weight weight{};
        weight.payload = storage.p;
        weight.payload_bytes = storage.bytes;
        weight.qtype = QType::GGML_Q8_0;
        weight.group_size = 32;
        weight.shape[0] = rows;
        weight.shape[1] = ops::kPleEmbeddingWidth;
        weight.padded_shape[0] = rows;
        weight.padded_shape[1] = ops::kPleEmbeddingWidth;
        weight.ndim = 2;
        weight.qdata = storage.p;
        weight.n = rows;
        weight.k = ops::kPleEmbeddingWidth;
        weight.group = 32;
        weight.layout = QuantLayout::GgmlBlockRow;
        return weight;
    }

    std::vector<std::uint8_t> key_host;
    std::vector<std::uint8_t> value_host;
    std::vector<float> source_key,source_value;
    DeviceBuffer key_device;
    DeviceBuffer value_device;
    std::uint16_t scale_bits{};
    float scale{};
    double key_unit{}, value_unit{};
    Weight key;
    Weight value;
};

struct Inputs {
    explicit Inputs(int width) : width(width), residual(static_cast<std::size_t>(ops::kPleChannels) * width),
                                 embedding(static_cast<std::size_t>(ops::kPleEmbeddingWidth) * width),
                                 conv_weight(static_cast<std::size_t>(ops::kPleChannels) * 4),
                                 norm(ops::kPleChannels, 0.75F),query_norm(norm),conv_norm(norm) {
        for (int token = 0; token < width; ++token) {
            embedding[ops::kPleEmbeddingWidth * token] = 1.0F + token * 0.125F;
            embedding[ops::kPleEmbeddingWidth * token + 1] = 0.5F + token * 0.125F;
            for (int branch = 0; branch < 4; ++branch) {
                const int agreement = (branch & 1) == 0 ? 1312 : 1248;
                for (int d = 0; d < ops::kPleEmbeddingWidth; ++d) {
                    const float key_sign = (d & 1) == 0 ? 1.0F : -1.0F;
                    residual[d + ops::kPleEmbeddingWidth * (branch + 4 * token)] =
                        d < agreement ? key_sign : -key_sign;
                }
            }
        }
        for (int tap = 0; tap < 4; ++tap) {
            for (int channel = 0; channel < ops::kPleChannels; ++channel) {
                conv_weight[static_cast<std::size_t>(channel) * 4 + tap] =
                    0.05F * (tap + 1) + 0.000001F * (channel % 17);
            }
        }
        round_to_bf16(residual);
        round_to_bf16(embedding);
    }

    int width;
    std::vector<float> residual;
    std::vector<float> embedding;
    std::vector<float> conv_weight;
    std::vector<float> norm;
    std::vector<float> query_norm,conv_norm;
    bool native_parameters=false;
};

struct RunResult {
    std::vector<std::uint16_t> output;
    std::vector<std::uint16_t> state;
    int guards = 0;
};

RunResult run_inject(std::span<const float> residual, std::span<const float> embedding, int width,
                     std::span<const std::uint16_t> old_state, const Inputs& inputs,
                     const Weights& weights, bool in_place_state,
                     bool in_place_residual = false, const Tensor* device_embedding = nullptr) {
    std::vector<float> residual_vec(residual.begin(), residual.end());
    std::vector<float> embedding_vec(embedding.begin(), embedding.end());
    auto dresidual = to_device_bf16(residual_vec);
    auto dembedding = device_embedding ? DeviceBuffer{} : to_device_bf16(embedding_vec);
    auto dnorm_key = inputs.native_parameters ? to_device_bf16(inputs.norm) : to_device_f32(inputs.norm);
    auto dnorm_query = inputs.native_parameters ? to_device_bf16(inputs.query_norm) : to_device_f32(inputs.query_norm);
    auto dnorm_conv = inputs.native_parameters ? to_device_bf16(inputs.conv_norm) : to_device_f32(inputs.conv_norm);
    auto dconv = inputs.native_parameters ? to_device_bf16(inputs.conv_weight) : to_device_f32(inputs.conv_weight);
    auto dold = to_device(std::vector<std::uint16_t>(old_state.begin(), old_state.end()));
    GuardedDeviceBuffer dnew(old_state.size_bytes());
    GuardedDeviceBuffer dout(residual.size()*sizeof(std::uint16_t));
    DeviceArena workspace(ops::ple_workspace_capacity_bytes(width, weights.key.qtype, weights.value.qtype));
    Tensor residual_t(dresidual.p, DType::BF16, {ops::kPleEmbeddingWidth, 4, width});
    Tensor embedding_t = device_embedding ? device_embedding->reshape({ops::kPleEmbeddingWidth,width}) :
        Tensor(dembedding.p, DType::BF16, {ops::kPleEmbeddingWidth, width});
    const auto parameter_type=inputs.native_parameters ? DType::BF16 : DType::FP32;
    Tensor key_norm_t(dnorm_key.p, parameter_type, {ops::kPleChannels});
    Tensor query_norm_t(dnorm_query.p, parameter_type, {ops::kPleChannels});
    Tensor conv_norm_t(dnorm_conv.p, parameter_type, {ops::kPleChannels});
    Tensor conv_t(dconv.p, parameter_type, {4, ops::kPleChannels});
    Tensor old_t(dold.p, DType::BF16, {ops::kPleChannels, 9});
    Tensor new_t(dnew.data(), DType::BF16, {ops::kPleChannels, 9});
    Tensor out_t(dout.data(), DType::BF16, {ops::kPleEmbeddingWidth, 4, width});
    Tensor& state_out = in_place_state ? old_t : new_t;
    Tensor& output = in_place_residual ? residual_t : out_t;
    ops::ple_inject(residual_t, embedding_t, weights.key, weights.value, key_norm_t,
                    query_norm_t, conv_norm_t, conv_t, old_t, state_out, output, workspace,
                    inputs.native_parameters ? ops::PleNormFormat::ZeroCenteredBf16 : ops::PleNormFormat::EffectiveFp32,
                    nullptr);
    cuda_synchronize();
    RunResult result;
    result.output = from_device<std::uint16_t>(in_place_residual ? dresidual.p : dout.data(),
                                               residual.size());
    result.state = from_device<std::uint16_t>(in_place_state ? dold.p : dnew.data(), old_state.size());
    if (!in_place_residual) { result.guards += dout.verify_guards("PLE injection output"); }
    if (!in_place_state) { result.guards += dnew.verify_guards("PLE new convolution state"); }
    return result;
}

struct OracleResult {
    std::vector<double> output;
    std::vector<std::uint16_t> state;
};

OracleResult oracle(const Inputs& input, const Weights& weights,
                    std::span<const std::uint16_t> initial_state) {
    constexpr int H=2560,C=10240;
    std::vector<double> history(initial_state.size()),current(C*input.width),output(C*input.width);
    std::transform(initial_state.begin(),initial_state.end(),history.begin(),bf16_to_f32);
    const auto gamma=[&](float value) { return double(value)+(input.native_parameters?1.0:0.0); };
    for(int t=0;t<input.width;++t) {
        // Complete ideal projection from represented inputs/weights, no private GEMM casts.
        const auto [key,value]=weights.project(std::span(input.embedding).subspan(H*t,H));
        std::array<double,4> gate{};
        for(int b=0;b<4;++b) {
            double q2=0,k2=0;
            for(int d=0;d<H;++d) {
                const double q=input.residual[C*t+H*b+d];
                q2+=q*q; k2+=key[H*b+d]*key[H*b+d];
            }
            const double qi=1/std::sqrt(q2/H+1e-6),ki=1/std::sqrt(k2/H+1e-6);
            double dot=0;
            for(int d=0;d<H;++d) {
                const int c=H*b+d;
                dot+=double(input.residual[C*t+c])*qi*gamma(input.query_norm[c])*
                     key[c]*ki*gamma(input.norm[c]);
            }
            const double z=dot/std::sqrt(double(H));
            const double transformed=z==0 ? 0 : std::copysign(std::sqrt(std::max(std::abs(z),1e-6)),z);
            gate[b]=1/(1+std::exp(-transformed));
            double g2=0;
            for(int d=0;d<H;++d) { const double g=gate[b]*value[d]; g2+=g*g; }
            const double gi=1/std::sqrt(g2/H+1e-6);
            for(int d=0;d<H;++d) {
                const int c=H*b+d;
                // This cast is the public persistent convolution-history boundary.
                current[C*t+c]=qwen4_sequence::represented(gate[b]*value[d]*gi*gamma(input.conv_norm[c]));
            }
        }
        for(int b=0;b<4;++b) for(int d=0;d<H;++d) {
            const int c=H*b+d;
            double conv=0;
            for(int tap=0;tap<4;++tap) {
                const int logical=t-9+3*tap;
                const double n=logical<0 ? history[C*(logical+9)+c] : current[C*logical+c];
                conv+=double(input.conv_weight[c*4+tap])*n;
            }
            output[C*t+c]=double(input.residual[C*t+c])+gate[b]*value[d]+conv/(1+std::exp(-conv));
        }
    }
    std::vector<std::uint16_t> final_state(initial_state.size());
    for(int h=0;h<9;++h) for(int c=0;c<C;++c) {
        const int t=input.width-9+h;
        final_state[C*h+c]=t<0 ? initial_state[C*(t+9)+c] : f32_to_bf16(current[C*t+c]);
    }
    return {std::move(output),std::move(final_state)};
}

int injection_state_case(QType type = QType::GGML_Q8_0, bool native_parameters = false) {
    constexpr int width = 4;
    Inputs input(width);
    if (native_parameters) {
        input.native_parameters=true;
        std::fill(input.norm.begin(),input.norm.end(),-.25F);
        input.query_norm=input.norm; input.conv_norm=input.norm;
        round_to_bf16(input.conv_weight);
    }
    Weights weights(type);
    std::vector<std::uint16_t> initial_state(static_cast<std::size_t>(ops::kPleChannels) * 9);
    for (int history = 0; history < 9; ++history) {
        for (int channel = 0; channel < ops::kPleChannels; ++channel) {
            const float magnitude =
                0.01F + 0.002F * history + 0.0001F * static_cast<float>(channel % 19);
            const float value = ((channel + history) & 1) == 0 ? magnitude : -magnitude;
            initial_state[static_cast<std::size_t>(history) * ops::kPleChannels + channel] =
                f32_to_bf16(value);
        }
    }
    const OracleResult expected = oracle(input, weights, initial_state);
    const RunResult one = run_inject(input.residual, input.embedding, width, initial_state, input,
                                     weights, false);
    int failures = one.guards;
    const auto actual = [&] {
        std::vector<double> values(one.output.size());
        for (std::size_t i = 0; i < values.size(); ++i) { values[i] = bf16_to_f32(one.output[i]); }
        return values;
    }();
    failures += verify_pointwise("PLE injection FP64 formula", actual, expected.output,
                                 PointwiseCriterion{0.02, 0.01});
    failures += verify_exact("PLE injection independent final BF16 state", one.state,
                             expected.state);
    const RunResult in_place = run_inject(input.residual, input.embedding, width, initial_state, input,
                                          weights, false, true);
    failures += verify_exact("PLE distinct vs in-place residual output", one.output,
                             in_place.output);
    failures += verify_exact("PLE distinct vs in-place residual state", one.state, in_place.state);
    failures += in_place.guards;

    const std::size_t channels = ops::kPleChannels;
    const RunResult first = run_inject(std::span(input.residual).first(2 * channels),
                                       std::span(input.embedding).first(2 * ops::kPleEmbeddingWidth),
                                       2, initial_state, input, weights, true);
    const RunResult second = run_inject(std::span(input.residual).subspan(2 * channels),
                                        std::span(input.embedding).subspan(2 * ops::kPleEmbeddingWidth),
                                        2, first.state, input, weights, false);
    std::vector<std::uint16_t> chunked = first.output;
    chunked.insert(chunked.end(), second.output.begin(), second.output.end());
    failures += verify_exact("PLE one-shot vs chunked output", one.output, chunked);
    failures += verify_exact("PLE one-shot vs chunked state", one.state, second.state);
    failures += first.guards + second.guards;

    std::vector<std::uint16_t> repeated_output;
    std::vector<std::uint16_t> repeated_state = initial_state;
    for (int token = 0; token < width; ++token) {
        const RunResult step = run_inject(
            std::span(input.residual).subspan(static_cast<std::size_t>(token) * channels, channels),
            std::span(input.embedding).subspan(static_cast<std::size_t>(token) * ops::kPleEmbeddingWidth,
                                                ops::kPleEmbeddingWidth),
            1, repeated_state, input, weights, true);
        repeated_output.insert(repeated_output.end(), step.output.begin(), step.output.end());
        repeated_state = step.state;
        failures += step.guards;
    }
    failures += verify_exact("PLE one-shot vs repeated T1 output", one.output, repeated_output);
    failures += verify_exact("PLE one-shot vs repeated T1 state", one.state, repeated_state);
    return failures;
}

std::vector<std::uint16_t> native_ple_bits(const artifact::Reader& reader, const std::string& role,
                                         std::vector<std::uint64_t> shape) {
    const auto* object=reader.find("model.language_model.layers.1.ple."+role);
    const auto* descriptor=object ? std::get_if<artifact::TensorDescriptor>(object) : nullptr;
    if(!descriptor || descriptor->format!=artifact::NumericFormat::BF16 ||
       descriptor->layout!=artifact::StorageLayout::ContiguousLeV1 || descriptor->shape!=shape) {
        throw std::runtime_error("invalid native PLE role "+role);
    }
    const auto payload=reader.payload(*object);
    std::vector<std::uint16_t> bits(payload.data.size()/2);
    for(std::size_t i=0;i<bits.size();++i) {
        bits[i]=std::to_integer<unsigned>(payload.data[2*i]) |
                (std::to_integer<unsigned>(payload.data[2*i+1])<<8);
    }
    return bits;
}

std::vector<double> represented_bf16(std::span<const std::uint16_t> bits) {
    std::vector<double> values(bits.size());
    std::transform(bits.begin(),bits.end(),values.begin(),bf16_to_f32);
    return values;
}

int native_ple_case(const std::string& root,
    const qwen4_sequence::Result* sequence_input=nullptr, qwen4_sequence::Result* sequence_output=nullptr,
    bool partitioned=false, bool nvfp4_table=false, bool text_panel=false) {
    if(text_panel && (!sequence_input || nvfp4_table)) {
        throw std::runtime_error("text PLE panel requires sequence and source FP8 table");
    }
    const auto text=text_panel?std::make_unique<qwen4_sequence::TextPanel>(root):nullptr;
    artifact::Reader reader(root+"/qwen4-ple-component.ninfer");
    if(reader.identity()!=artifact::ArtifactIdentity{
        "qwen4/native-ple-component-qualification","nvidia-bf16-source"}) {
        throw std::runtime_error("invalid native PLE component identity");
    }
    Weights weights(native_ple_bits(reader,"key_proj.weight",{10240,2560}),
                    native_ple_bits(reader,"value_proj.weight",{2560,2560}));
    const auto values=[&](const std::string& role,std::vector<std::uint64_t> shape) {
        const auto bits=native_ple_bits(reader,role,std::move(shape));
        std::vector<float> result(bits.size());
        std::transform(bits.begin(),bits.end(),result.begin(),bf16_to_f32);
        return result;
    };
    const auto key_norm=values("norm_key.weight",{10240}),query_norm=values("norm_query.weight",{10240}),
        conv_norm=values("norm_conv.weight",{10240}),conv=values("conv1d.weight",{10240,1,4});
    int failures=0;
    // Same ideal formula and fixed criteria for both table representations. These are
    // ordinal fixture rows, not hashes for a text sequence and not a model quality benchmark.
    constexpr ReductionCriterion output_gate{0.02,1e-4,0.02};
    constexpr ReductionCriterion state_gate{1.0/256,1e-5,1.0/128};
    for(bool nvfp4:{false,true}) {
        if(sequence_input && nvfp4!=nvfp4_table) { continue; }
        artifact::Reader table_reader(root+(text_panel?"/qwen4-text-panel.ninfer":
            nvfp4?"/qwen4-ple-nvfp4-rows.ninfer":"/qwen4-ple-rows.ninfer"));
        const artifact::ArtifactIdentity expected_identity=text_panel
            ? artifact::ArtifactIdentity{"qwen4/native-text-qualification","nvidia-source-33"}
            : artifact::ArtifactIdentity{"qwen4/native-ple-qualification",
                nvfp4?"primitive-nvfp4-source-rows":"nvidia-fp8-source-rows"};
        if(table_reader.identity()!=expected_identity) {
            throw std::runtime_error("invalid native PLE table identity");
        }
        artifact::Binder binder(table_reader);
        const auto handle=nvfp4
            ? artifact::bind_tensor(binder,"ple.rows",artifact::NumericFormat::NVFP4_PARTITION_F32M,
                {3,4,160},artifact::TensorPlacement::ResidentHost)
            : artifact::bind_tensor(binder,"ple.rows",artifact::NumericFormat::FP8_E4M3FN_TENSOR_BF16S,
                {text?text->row_count:16,160},artifact::TensorPlacement::ResidentHost);
        if(text) {
            (void)artifact::bind_tensor(binder,"token.embeddings",artifact::NumericFormat::BF16,
                {33,2560},artifact::TensorPlacement::ValidateOnly);
            (void)artifact::bind_tensor(binder,"token.ids",artifact::NumericFormat::I32,
                {33},artifact::TensorPlacement::ValidateOnly);
            for(const char* name:{"ple.global_rows","ple.local_rows"}) {
                (void)artifact::bind_tensor(binder,name,artifact::NumericFormat::I32,
                    {33,16},artifact::TensorPlacement::ValidateOnly);
            }
        }
        DeviceContext device(0);
        auto resident=artifact::materialize(table_reader,binder.finish(),device);
        const auto bytes=resident.mapped_tensor_bytes(handle);
        const auto* table=reinterpret_cast<const std::uint8_t*>(bytes.data());
        if(resident.stats().resident_locked_bytes<bytes.size()) { throw std::runtime_error("PLE sample not locked"); }
        const int rows=text?static_cast<int>(text->row_count):nvfp4?12:16,record_bytes=nvfp4?94:160;
        const std::uint16_t scale=nvfp4?0:table[rows*160]|(table[rows*160+1]<<8);
        const std::vector<int> widths=sequence_input
            ? std::vector<int>{static_cast<int>(sequence_input->actual.size()/10240)}
            : std::vector<int>{1,9,10,17,27,28,29};
        for(int width:widths) {
            std::cout<<"native PLE table="<<(nvfp4?"nvfp4":"fp8")<<" T="<<width<<'\n';
            Inputs input(width);
            input.native_parameters=true; input.norm=key_norm; input.query_norm=query_norm;
            input.conv_norm=conv_norm; input.conv_weight=conv;
            fill_uniform(input.residual,88921U,-.3F,.3F); round_to_bf16(input.residual);
            if(sequence_input) { input.residual=sequence_input->actual; }
            std::vector<int> ids(16*width);
            std::vector<std::uint16_t> expected_embedding(width*2560);
            for(int i=0;i<width*16;++i) {
                ids[i]=text?text->local_rows.at(i):(i*7+i/16)%rows;
                if(nvfp4) {
                    std::array<std::uint8_t,94> record{};
                    std::copy_n(table+ids[i]*90,90,record.begin());
                    std::copy_n(table+1080+(ids[i]/4)*4,4,record.begin()+90);
                    for(int d=0;d<160;++d) { expected_embedding[i*160+d]=ple_nvfp4_oracle(record.data(),d); }
                } else {
                    for(int d=0;d<160;++d) {
                        expected_embedding[i*160+d]=fp8_ple_oracle(table[ids[i]*160+d],scale);
                    }
                }
            }
            std::transform(expected_embedding.begin(),expected_embedding.end(),input.embedding.begin(),bf16_to_f32);
            PinnedHostBuffer pinned(width*16*record_bytes);
            GuardedDeviceBuffer packed(pinned.size()),embedding(expected_embedding.size()*2);
            Tensor encoded(packed.data(),DType::U8,{record_bytes,16,width});
            Tensor decoded(embedding.data(),DType::BF16,{160,16,width});
            if(nvfp4) {
                ops::ple_nvfp4_stage_rows_batch({table,3,4,bytes.size()},ids,width,
                    pinned.data(),pinned.size(),encoded,nullptr);
                ops::ple_nvfp4_decode_rows(encoded,decoded,nullptr);
            } else {
                ops::ple_fp8_stage_rows_batch({table,static_cast<std::size_t>(rows),static_cast<std::size_t>(rows)*160},ids,width,
                    pinned.data(),pinned.size(),encoded,nullptr);
                ops::ple_fp8_decode_rows(encoded,scale,decoded,nullptr);
            }
            // Do not read GPU embedding to construct the math oracle. It comes from independent
            // source-code decode above; the real GPU result directly feeds the public injection.
            std::vector<std::uint16_t> initial(10240*9);
            if(!sequence_input) {
                for(std::size_t i=0;i<initial.size();++i) { initial[i]=f32_to_bf16(float(int(i%31)-15)/128); }
            }
            const auto expected=oracle(input,weights,initial);
            const auto whole=run_inject(input.residual,input.embedding,width,initial,input,weights,false,false,&decoded);
            auto selected_output=whole.output;
            failures+=whole.guards;
            failures+=verify_exact("native PLE packed-to-BF16 boundary",
                from_device<std::uint16_t>(embedding.data(),expected_embedding.size()),expected_embedding);
            failures+=verify_reduction("native PLE complete FP64 output",represented_bf16(whole.output),expected.output,output_gate);
            failures+=verify_reduction("native PLE complete FP64 state",represented_bf16(whole.state),
                represented_bf16(expected.state),state_gate);
            if(width<9) {
                failures+=verify_exact("native PLE untouched history prefix",
                    std::vector<std::uint16_t>(whole.state.begin(),whole.state.begin()+(9-width)*10240),
                    std::vector<std::uint16_t>(initial.begin()+width*10240,initial.end()));
            }
            if((!sequence_input && (width==10 || width==17 || width==29)) || (sequence_input && partitioned)) {
                std::vector<std::uint16_t> actual_state=initial,chunked;
                for(int begin=0;begin<width;) {
                    const int count=std::min(begin==0?std::max(1,width-1):1,width-begin);
                    Inputs chunk(count);
                    chunk.native_parameters=true;chunk.norm=key_norm;chunk.query_norm=query_norm;
                    chunk.conv_norm=conv_norm;chunk.conv_weight=conv;
                    std::copy_n(input.residual.begin()+begin*10240,count*10240,chunk.residual.begin());
                    std::copy_n(input.embedding.begin()+begin*2560,count*2560,chunk.embedding.begin());
                    const auto ref=oracle(chunk,weights,actual_state);
                    auto device_chunk=decoded.slice(2,begin,count);
                    const auto result=run_inject(chunk.residual,chunk.embedding,count,actual_state,chunk,weights,true,true,&device_chunk);
                    failures+=result.guards;
                    failures+=verify_reduction("native PLE continued FP64 output",represented_bf16(result.output),ref.output,output_gate);
                    failures+=verify_reduction("native PLE continued FP64 state",represented_bf16(result.state),
                        represented_bf16(ref.state),state_gate);
                    chunked.insert(chunked.end(),result.output.begin(),result.output.end());
                    actual_state=result.state;begin+=count;
                }
                failures+=verify_reduction("native PLE accumulated chunk FP64 output",represented_bf16(chunked),expected.output,output_gate);
                failures+=verify_reduction("native PLE accumulated chunk FP64 state",represented_bf16(actual_state),represented_bf16(expected.state),state_gate);
                failures+=verify_reduction("native PLE whole/chunk output",represented_bf16(chunked),represented_bf16(whole.output),output_gate);
                failures+=verify_reduction("native PLE whole/chunk state",represented_bf16(actual_state),represented_bf16(whole.state),state_gate);
                if(sequence_input && partitioned) { selected_output=std::move(chunked); }
            }
            if(!sequence_input && width==9) {
                // Reset removes the previous represented history; reuse is not an implicit cache.
                std::vector<std::uint16_t> zero(initial.size());
                const auto ref=oracle(input,weights,zero);
                const auto reset=run_inject(input.residual,input.embedding,width,zero,input,weights,true,true,&decoded);
                failures+=reset.guards;
                failures+=verify_reduction("native PLE zero-state reset output",represented_bf16(reset.output),ref.output,output_gate);
                failures+=verify_reduction("native PLE zero-state reset history",represented_bf16(reset.state),
                    represented_bf16(ref.state),state_gate);
            }
            failures+=packed.verify_guards("native PLE gathered payload");
            failures+=embedding.verify_guards("native PLE embedding");
            if(sequence_output) {
                sequence_output->actual.resize(selected_output.size());
                std::transform(selected_output.begin(),selected_output.end(),sequence_output->actual.begin(),bf16_to_f32);
                Inputs independent_input=input;
                independent_input.residual=sequence_input->reference;
                const auto reference=oracle(independent_input,weights,initial);
                sequence_output->reference=qwen4_sequence::represented(reference.output);
            }
        }
    }
    return failures;
}

} // namespace

#ifdef NINFER_QWEN4_SEQUENCE_COMPONENTS
namespace ninfer::test::qwen4_sequence {
Result ple(const std::string& root,const Result& input,bool partitioned,bool nvfp4_table,bool text_panel) {
    Result output;
    output.failures=native_ple_case(root,&input,&output,partitioned,nvfp4_table,text_panel);
    return output;
}
}
#else
int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    if (argc == 2 && std::string_view(argv[1]) == "--native-real") {
        const char* root = std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
        if (!root) { return 77; }
        const int failures = native_ple_case(root);
        std::cout << (failures ? "FAIL" : "PASS") << " native PLE complete FP64 oracle\n";
        return failures ? 1 : 0;
    }
    static_assert(ops::kPleMaxStagedBytes == 5'898'240);
    int failures = 0;
    for (const int width : {1, 17, 128, 4096}) { failures += nvfp4_staging_decode_case(width); }
    for (const int width : {1, 17, 128, 4096}) { failures += fp8_staging_decode_case(width); }
    failures += staging_decode_case();
    for (const int width : {3, 16, 17, 128, 4096}) {
        failures += batched_staging_decode_case(width);
    }
    failures += injection_state_case();
    failures += injection_state_case(QType::BF16_CTRL);
    failures += injection_state_case(QType::BF16_CTRL,true);
    failures += injection_state_case(QType::NVFP4);
    failures += injection_state_case(QType::FP8_E4M3FN_ROW_BF16S);
    if (failures != 0) {
        std::cerr << "PLE tests failed: " << failures << '\n';
        return 1;
    }
    std::cout << "PLE tests passed\n";
    return 0;
}
#endif
