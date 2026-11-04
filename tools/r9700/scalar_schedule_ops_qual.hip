#include "core/tensor.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/nll_from_logits.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/types.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::int32_t kVocabPhysical  = 248320;
constexpr std::int32_t kVocabValid     = 248077;
constexpr std::int32_t kDflashFeatures = 25600;
constexpr std::int32_t kDflashWidth    = 16;
constexpr std::int32_t kMaxBatch = static_cast<std::int32_t>(ninfer::kDFlashMaximumConcurrency);
constexpr std::size_t kGuardBytes      = 256;
constexpr std::uint8_t kGuard          = 0xA7;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void hip_check(hipError_t status, const char* expression, const char* file, int line) {
    if (status == hipSuccess) { return; }
    std::ostringstream message;
    message << file << ':' << line << ": " << expression << ": " << hipGetErrorString(status);
    fail(message.str());
}

#define QUAL_HIP_CHECK(expression) hip_check((expression), #expression, __FILE__, __LINE__)

class GuardedDeviceBytes {
public:
    explicit GuardedDeviceBytes(std::size_t bytes) : bytes_(bytes) {
        QUAL_HIP_CHECK(hipMalloc(&storage_, bytes_ + 2 * kGuardBytes));
        QUAL_HIP_CHECK(hipMemset(storage_, kGuard, bytes_ + 2 * kGuardBytes));
    }

    GuardedDeviceBytes(const GuardedDeviceBytes&)            = delete;
    GuardedDeviceBytes& operator=(const GuardedDeviceBytes&) = delete;

    ~GuardedDeviceBytes() { (void)hipFree(storage_); }

    [[nodiscard]] void* data() const noexcept {
        return static_cast<std::byte*>(storage_) + kGuardBytes;
    }

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    template <typename T>
    void upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != bytes_) { fail("guarded upload extent mismatch"); }
        QUAL_HIP_CHECK(hipMemcpy(data(), values.data(), bytes_, hipMemcpyHostToDevice));
    }

    template <typename T>
    [[nodiscard]] std::vector<T> download() const {
        if ((bytes_ % sizeof(T)) != 0) { fail("guarded download element mismatch"); }
        std::vector<T> values(bytes_ / sizeof(T));
        QUAL_HIP_CHECK(hipMemcpy(values.data(), data(), bytes_, hipMemcpyDeviceToHost));
        return values;
    }

    void fill(std::uint8_t value) { QUAL_HIP_CHECK(hipMemset(data(), value, bytes_)); }

    void verify_guards(const char* label) const {
        std::vector<std::uint8_t> prefix(kGuardBytes);
        std::vector<std::uint8_t> suffix(kGuardBytes);
        QUAL_HIP_CHECK(hipMemcpy(prefix.data(), storage_, kGuardBytes, hipMemcpyDeviceToHost));
        QUAL_HIP_CHECK(hipMemcpy(suffix.data(), static_cast<const std::byte*>(data()) + bytes_,
                                 kGuardBytes, hipMemcpyDeviceToHost));
        if (!std::all_of(prefix.begin(), prefix.end(),
                         [](std::uint8_t value) { return value == kGuard; }) ||
            !std::all_of(suffix.begin(), suffix.end(),
                         [](std::uint8_t value) { return value == kGuard; })) {
            fail(std::string(label) + " guard modified");
        }
    }

private:
    void* storage_     = nullptr;
    std::size_t bytes_ = 0;
};

template <typename T>
void require_exact(const std::vector<T>& actual, const std::vector<T>& expected,
                   const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent differs"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            fail(std::string(label) + " mismatch at " + std::to_string(index));
        }
    }
}

void require_bf16_bits(const std::vector<hip_bfloat16>& actual,
                       const std::vector<hip_bfloat16>& expected, const char* label) {
    if (actual.size() != expected.size()) { fail(std::string(label) + " extent differs"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].data != expected[index].data) {
            fail(std::string(label) + " BF16 bits differ at " + std::to_string(index));
        }
    }
}

void select_r9700() {
    int count = 0;
    QUAL_HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        QUAL_HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        if (properties.integrated == 0 &&
            std::string_view(properties.gcnArchName).starts_with("gfx1201")) {
            if (selected >= 0) { fail("expected exactly one discrete gfx1201 device"); }
            selected = ordinal;
        }
    }
    if (selected < 0) { fail("no discrete gfx1201 R9700 found"); }
    QUAL_HIP_CHECK(hipSetDevice(selected));
    hipDeviceProp_t properties{};
    QUAL_HIP_CHECK(hipGetDeviceProperties(&properties, selected));
    if (properties.warpSize != 32) { fail("scalar/schedule qualification requires wave32"); }
    std::cout << "device arch=" << properties.gcnArchName << " wave=" << properties.warpSize
              << '\n';
}

template <typename Launch>
double timed_us(hipStream_t stream, int warmup, int iterations, Launch&& launch) {
    hipEvent_t begin = nullptr;
    hipEvent_t end   = nullptr;
    QUAL_HIP_CHECK(hipEventCreate(&begin));
    QUAL_HIP_CHECK(hipEventCreate(&end));
    for (int iteration = 0; iteration < warmup; ++iteration) { launch(); }
    QUAL_HIP_CHECK(hipEventRecord(begin, stream));
    for (int iteration = 0; iteration < iterations; ++iteration) { launch(); }
    QUAL_HIP_CHECK(hipEventRecord(end, stream));
    QUAL_HIP_CHECK(hipEventSynchronize(end));
    float milliseconds = 0.0F;
    QUAL_HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
    QUAL_HIP_CHECK(hipEventDestroy(begin));
    QUAL_HIP_CHECK(hipEventDestroy(end));
    return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

std::vector<hip_bfloat16> bf16_fixture(std::size_t count, std::uint32_t seed, float scale = 1.0F) {
    std::vector<hip_bfloat16> values(count);
    std::uint32_t state = seed;
    for (std::size_t index = 0; index < count; ++index) {
        state            = state * 1664525U + 1013904223U;
        const float unit = static_cast<float>((state >> 8U) & 0xffffU) / 32768.0F - 1.0F;
        values[index]    = hip_bfloat16(unit * scale);
    }
    return values;
}

void qualify_add_bias(hipStream_t stream) {
    struct Shape {
        std::int32_t features;
        std::int32_t rows;
    };

    constexpr Shape shapes[]{{1152, 1}, {3456, 64}, {4304, 257}, {1153, 3}};
    double production_us = 0.0;
    for (const Shape shape : shapes) {
        auto bias  = bf16_fixture(shape.features, 17U + shape.features, 2.0F);
        auto input = bf16_fixture(static_cast<std::size_t>(shape.features) * shape.rows,
                                  71U + shape.rows, 8.0F);
        std::vector<double> oracle(input.size());
        for (std::size_t index = 0; index < input.size(); ++index) {
            oracle[index] = static_cast<double>(static_cast<float>(input[index])) +
                            static_cast<double>(static_cast<float>(bias[index % bias.size()]));
        }
        GuardedDeviceBytes d_bias(bias.size() * sizeof(hip_bfloat16));
        GuardedDeviceBytes d_input(input.size() * sizeof(hip_bfloat16));
        d_bias.upload(bias);
        d_input.upload(input);
        ninfer::Tensor bias_tensor(d_bias.data(), ninfer::DType::BF16, {shape.features});
        ninfer::Tensor input_tensor(d_input.data(), ninfer::DType::BF16,
                                    {shape.features, shape.rows});
        ninfer::ops::add_bias(bias_tensor, input_tensor, stream);
        QUAL_HIP_CHECK(hipStreamSynchronize(stream));
        const auto actual = d_input.download<hip_bfloat16>();
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const double observed = static_cast<double>(static_cast<float>(actual[index]));
            const double absolute = std::abs(observed - oracle[index]);
            const double relative = absolute / std::max(1.0, std::abs(oracle[index]));
            if (absolute > 0.0625 && relative > 0.004) {
                fail("add_bias FP64 oracle mismatch at " + std::to_string(index));
            }
        }
        require_bf16_bits(d_bias.download<hip_bfloat16>(), bias, "add_bias immutable bias");
        d_bias.verify_guards("add_bias bias");
        d_input.verify_guards("add_bias output");
        if (shape.features == 4304) {
            production_us = timed_us(
                stream, 10, 100, [&] { ninfer::ops::add_bias(bias_tensor, input_tensor, stream); });
        }
    }
    std::cout << "add_bias PASS D4304x257_us=" << production_us << '\n';
}

void qualify_nll(hipStream_t stream) {
    constexpr std::int32_t columns = 64;
    auto logits = bf16_fixture(static_cast<std::size_t>(kVocabPhysical) * columns, 0x9137U, 8.0F);
    std::vector<std::int32_t> targets(columns);
    std::vector<double> oracle(columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * kVocabPhysical;
        targets[column]        = (column * 7919 + 17) % kVocabValid;
        double maximum         = -std::numeric_limits<double>::infinity();
        for (std::int32_t row = 0; row < kVocabValid; ++row) {
            maximum =
                std::max(maximum, static_cast<double>(static_cast<float>(logits[base + row])));
        }
        double sum = 0.0;
        for (std::int32_t row = 0; row < kVocabValid; ++row) {
            sum += std::exp(static_cast<double>(static_cast<float>(logits[base + row])) - maximum);
        }
        oracle[column] = maximum + std::log(sum) -
                         static_cast<double>(static_cast<float>(logits[base + targets[column]]));
        logits[base + kVocabValid] = hip_bfloat16(64.0F); // padded row must not participate
    }
    GuardedDeviceBytes d_logits(logits.size() * sizeof(hip_bfloat16));
    GuardedDeviceBytes d_targets(targets.size() * sizeof(std::int32_t));
    GuardedDeviceBytes d_output(oracle.size() * sizeof(float));
    d_logits.upload(logits);
    d_targets.upload(targets);
    d_output.fill(0xCD);
    ninfer::Tensor logits_tensor(d_logits.data(), ninfer::DType::BF16, {kVocabPhysical, columns});
    ninfer::Tensor targets_tensor(d_targets.data(), ninfer::DType::I32, {columns});
    ninfer::Tensor output_tensor(d_output.data(), ninfer::DType::FP32, {columns});
    ninfer::ops::nll_from_logits(logits_tensor, targets_tensor, output_tensor, kVocabValid, stream);
    QUAL_HIP_CHECK(hipStreamSynchronize(stream));
    const auto actual = d_output.download<float>();
    double max_abs    = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        max_abs = std::max(max_abs, std::abs(static_cast<double>(actual[index]) - oracle[index]));
    }
    if (max_abs > 0.005) { fail("nll_from_logits exceeds FP64-oracle tolerance"); }
    require_bf16_bits(d_logits.download<hip_bfloat16>(), logits, "nll immutable logits");
    require_exact(d_targets.download<std::int32_t>(), targets, "nll immutable targets");
    d_logits.verify_guards("nll logits");
    d_targets.verify_guards("nll targets");
    d_output.verify_guards("nll output");
    const double elapsed = timed_us(stream, 5, 30, [&] {
        ninfer::ops::nll_from_logits(logits_tensor, targets_tensor, output_tensor, kVocabValid,
                                     stream);
    });
    std::cout << "nll_from_logits PASS V248077x64_max_abs=" << max_abs << " time_us=" << elapsed
              << '\n';
}

void qualify_masked_block(hipStream_t stream) {
    GuardedDeviceBytes d_anchors(kMaxBatch * sizeof(std::int32_t));
    GuardedDeviceBytes d_lengths(kMaxBatch * sizeof(std::int32_t));
    GuardedDeviceBytes d_valid(kMaxBatch * sizeof(std::int32_t));
    GuardedDeviceBytes d_ids(16 * kMaxBatch * sizeof(std::int32_t));
    GuardedDeviceBytes d_positions(16 * kMaxBatch * sizeof(std::int32_t));
    for (std::int32_t width = 1; width <= 16; ++width) {
        for (std::int32_t batch = 1; batch <= kMaxBatch; ++batch) {
            std::vector<std::int32_t> anchors(kMaxBatch);
            std::vector<std::int32_t> lengths(kMaxBatch);
            std::vector<std::int32_t> valid(kMaxBatch);
            for (std::int32_t row = 0; row < kMaxBatch; ++row) {
                anchors[row] = 1000 + width * 31 + row;
                lengths[row] = row == batch - 1
                                   ? std::numeric_limits<std::int32_t>::max() - (width - 1)
                                   : 27 + row * 101;
                valid[row]   = 1 + ((row * 5 + width) % width);
            }
            d_anchors.upload(anchors);
            d_lengths.upload(lengths);
            d_valid.upload(valid);
            d_ids.fill(0xCD);
            d_positions.fill(0xEF);
            ninfer::Tensor t_anchors(d_anchors.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor t_lengths(d_lengths.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor t_valid(d_valid.data(), ninfer::DType::I32, {batch});
            ninfer::Tensor t_ids(d_ids.data(), ninfer::DType::I32, {width, batch});
            ninfer::Tensor t_positions(d_positions.data(), ninfer::DType::I32, {width, batch});
            ninfer::ops::prepare_masked_block(t_anchors, t_lengths, t_valid, kVocabValid, t_ids,
                                              t_positions, stream);
            QUAL_HIP_CHECK(hipStreamSynchronize(stream));
            const auto ids              = d_ids.download<std::int32_t>();
            const auto positions        = d_positions.download<std::int32_t>();
            const auto logical_elements = static_cast<std::size_t>(width * batch);
            for (std::int32_t row = 0; row < batch; ++row) {
                for (std::int32_t column = 0; column < width; ++column) {
                    const std::size_t index        = static_cast<std::size_t>(row * width + column);
                    const std::int32_t expected_id = column == 0 ? anchors[row] : kVocabValid;
                    const std::int32_t expected_position =
                        lengths[row] + std::min(column, valid[row] - 1);
                    if (ids[index] != expected_id || positions[index] != expected_position) {
                        fail("prepare_masked_block exact oracle mismatch W=" +
                             std::to_string(width) + " B=" + std::to_string(batch));
                    }
                }
            }
            const std::int32_t ids_sentinel       = static_cast<std::int32_t>(0xCDCDCDCDU);
            const std::int32_t positions_sentinel = static_cast<std::int32_t>(0xEFEFEFEFU);
            for (std::size_t index = logical_elements; index < ids.size(); ++index) {
                if (ids[index] != ids_sentinel || positions[index] != positions_sentinel) {
                    fail("prepare_masked_block wrote beyond logical output");
                }
            }
            require_exact(d_anchors.download<std::int32_t>(), anchors, "masked immutable anchors");
            require_exact(d_lengths.download<std::int32_t>(), lengths, "masked immutable lengths");
            require_exact(d_valid.download<std::int32_t>(), valid,
                          "masked immutable valid columns");
        }
    }
    ninfer::Tensor t_anchors(d_anchors.data(), ninfer::DType::I32, {kMaxBatch});
    ninfer::Tensor t_lengths(d_lengths.data(), ninfer::DType::I32, {kMaxBatch});
    ninfer::Tensor t_valid(d_valid.data(), ninfer::DType::I32, {kMaxBatch});
    ninfer::Tensor t_ids(d_ids.data(), ninfer::DType::I32, {16, kMaxBatch});
    ninfer::Tensor t_positions(d_positions.data(), ninfer::DType::I32, {16, kMaxBatch});
    const double elapsed = timed_us(stream, 10, 200, [&] {
        ninfer::ops::prepare_masked_block(t_anchors, t_lengths, t_valid, kVocabValid, t_ids,
                                          t_positions, stream);
    });
    d_anchors.verify_guards("masked anchors");
    d_lengths.verify_guards("masked lengths");
    d_valid.verify_guards("masked valid");
    d_ids.verify_guards("masked ids");
    d_positions.verify_guards("masked positions");
    std::cout << "prepare_masked_block PASS W1..16_B1..8 W16B4_us=" << elapsed << '\n';
}

std::vector<hip_bfloat16> bit_fixture(std::size_t count, std::uint32_t seed) {
    std::vector<hip_bfloat16> values(count);
    std::uint32_t state = seed;
    for (std::size_t index = 0; index < count; ++index) {
        state              = state * 1103515245U + 12345U;
        values[index].data = static_cast<std::uint16_t>((state >> 16U) ^ index);
    }
    return values;
}

void qualify_ragged_prefix(hipStream_t stream) {
    constexpr std::int32_t capacity = kMaxBatch;
    const std::size_t source_count =
        static_cast<std::size_t>(kDflashFeatures) * kDflashWidth * capacity;
    const std::size_t output_count =
        static_cast<std::size_t>(kDflashFeatures) * kDflashWidth * kMaxBatch;
    const auto source = bit_fixture(source_count, 0x51EDC0DEU);
    // A fixed non-identity lane permutation and distinct starts for every row.
    std::vector<std::int32_t> lanes(kMaxBatch), starts(kMaxBatch);
    for (std::int32_t row = 0; row < kMaxBatch; ++row) {
        lanes[row]  = (row * 3 + kMaxBatch - 1) % kMaxBatch;
        starts[row] = 100 + row * 101;
    }
    std::vector<std::int32_t> ends(kMaxBatch);
    for (std::int32_t row = 0; row < kMaxBatch; ++row) {
        ends[row] = starts[row] + (row == kMaxBatch - 1 ? kDflashWidth : row);
    }
    std::vector<hip_bfloat16> expected(output_count);
    std::vector<std::int32_t> expected_positions(kDflashWidth * kMaxBatch);
    std::vector<std::int32_t> expected_counts(kMaxBatch);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        const std::int32_t count = ends[batch] - starts[batch];
        expected_counts[batch]   = count;
        for (std::int32_t column = 0; column < kDflashWidth; ++column) {
            const std::size_t out_base =
                static_cast<std::size_t>(batch * kDflashWidth + column) * kDflashFeatures;
            if (column < count) {
                const std::size_t source_base =
                    static_cast<std::size_t>(lanes[batch] * kDflashWidth + column) *
                    kDflashFeatures;
                std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(source_base),
                            kDflashFeatures,
                            expected.begin() + static_cast<std::ptrdiff_t>(out_base));
            }
            expected_positions[batch * kDflashWidth + column] =
                starts[batch] + (column < count ? column : std::max(count - 1, 0));
        }
    }
    GuardedDeviceBytes d_source(source.size() * sizeof(hip_bfloat16));
    GuardedDeviceBytes d_lanes(lanes.size() * sizeof(std::int32_t));
    GuardedDeviceBytes d_starts(starts.size() * sizeof(std::int32_t));
    GuardedDeviceBytes d_ends(ends.size() * sizeof(std::int32_t));
    GuardedDeviceBytes d_output(expected.size() * sizeof(hip_bfloat16));
    GuardedDeviceBytes d_positions(expected_positions.size() * sizeof(std::int32_t));
    GuardedDeviceBytes d_counts(expected_counts.size() * sizeof(std::int32_t));
    d_source.upload(source);
    d_lanes.upload(lanes);
    d_starts.upload(starts);
    d_ends.upload(ends);
    d_output.fill(0xCD);
    d_positions.fill(0xEF);
    d_counts.fill(0xAB);
    ninfer::Tensor t_source(d_source.data(), ninfer::DType::BF16,
                            {kDflashFeatures, kDflashWidth, capacity});
    ninfer::Tensor t_lanes(d_lanes.data(), ninfer::DType::I32, {kMaxBatch});
    ninfer::Tensor t_starts(d_starts.data(), ninfer::DType::I32, {kMaxBatch});
    ninfer::Tensor t_ends(d_ends.data(), ninfer::DType::I32, {kMaxBatch});
    ninfer::Tensor t_output(d_output.data(), ninfer::DType::BF16,
                            {kDflashFeatures, kDflashWidth, kMaxBatch});
    ninfer::Tensor t_positions(d_positions.data(), ninfer::DType::I32, {kDflashWidth, kMaxBatch});
    ninfer::Tensor t_counts(d_counts.data(), ninfer::DType::I32, {kMaxBatch});
    ninfer::ops::prepare_ragged_prefix(t_source, t_lanes, t_starts, t_ends, t_output, t_positions,
                                       t_counts, stream);
    QUAL_HIP_CHECK(hipStreamSynchronize(stream));
    require_bf16_bits(d_output.download<hip_bfloat16>(), expected, "ragged prefix values");
    require_exact(d_positions.download<std::int32_t>(), expected_positions,
                  "ragged prefix positions");
    require_exact(d_counts.download<std::int32_t>(), expected_counts, "ragged prefix counts");
    require_bf16_bits(d_source.download<hip_bfloat16>(), source, "ragged immutable source");
    require_exact(d_lanes.download<std::int32_t>(), lanes, "ragged immutable lanes");
    require_exact(d_starts.download<std::int32_t>(), starts, "ragged immutable starts");
    require_exact(d_ends.download<std::int32_t>(), ends, "ragged immutable ends");
    const double elapsed = timed_us(stream, 5, 30, [&] {
        ninfer::ops::prepare_ragged_prefix(t_source, t_lanes, t_starts, t_ends, t_output,
                                           t_positions, t_counts, stream);
    });
    d_source.verify_guards("ragged source");
    d_lanes.verify_guards("ragged lanes");
    d_starts.verify_guards("ragged starts");
    d_ends.verify_guards("ragged ends");
    d_output.verify_guards("ragged output");
    d_positions.verify_guards("ragged positions");
    d_counts.verify_guards("ragged counts");
    std::cout << "prepare_ragged_prefix PASS D25600_W16_B4_us=" << elapsed << '\n';
}

} // namespace

int main() {
    try {
        select_r9700();
        hipStream_t stream = nullptr;
        QUAL_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        qualify_add_bias(stream);
        qualify_nll(stream);
        qualify_masked_block(stream);
        qualify_ragged_prefix(stream);
        QUAL_HIP_CHECK(hipStreamDestroy(stream));
        std::cout << "scalar_schedule_ops: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "scalar_schedule_ops: FAIL: " << error.what() << '\n';
        return 1;
    }
}
