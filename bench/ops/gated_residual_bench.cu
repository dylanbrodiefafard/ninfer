// Public-Op benchmark for the fixed Qwen4-preview C=1 gated-residual read paths.
#include "ninfer/ops/gated_residual.h"
#include "ninfer_bench_common.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kBranches = 4;
constexpr std::int32_t kFlat = kHidden * kBranches;
constexpr std::int32_t kRank = 320;
constexpr std::size_t kQ8BlockBytes = 34;
constexpr double kDownBytes = static_cast<double>(kRank) * (kFlat / 32) * kQ8BlockBytes;
constexpr double kUpBytes = static_cast<double>(kFlat) * (kRank / 32) * kQ8BlockBytes;
constexpr double kReadInterfaceBytes =
    kDownBytes + kUpBytes + kFlat * sizeof(std::uint16_t) + kFlat * sizeof(float) +
    kHidden * sizeof(std::uint16_t);
constexpr double kReadWriteInterfaceBytes =
    kReadInterfaceBytes + static_cast<double>(kBranches) * kFlat * sizeof(float) +
    kBranches * sizeof(std::uint16_t);

enum class Operation { Both, Read, ReadWrite };

struct Options {
    Operation operation = Operation::Both;
    int warmup = 40;
    int repeat = 200;
    int min_time_ms = 1500;
    int profile_iterations = 0;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* name) {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + name); }
            return argv[i];
        };
        if (!std::strcmp(argv[i], "--operation")) {
            const std::string value = next("operation");
            if (value == "both") {
                options.operation = Operation::Both;
            } else if (value == "read") {
                options.operation = Operation::Read;
            } else if (value == "read-write") {
                options.operation = Operation::ReadWrite;
            } else {
                throw std::invalid_argument("operation must be both, read, or read-write");
            }
        } else if (!std::strcmp(argv[i], "--warmup")) {
            options.warmup = std::atoi(next("warmup"));
        } else if (!std::strcmp(argv[i], "--repeat")) {
            options.repeat = std::atoi(next("repeat"));
        } else if (!std::strcmp(argv[i], "--min-ms")) {
            options.min_time_ms = std::atoi(next("min-ms"));
        } else if (!std::strcmp(argv[i], "--profile-iterations")) {
            options.profile_iterations = std::atoi(next("profile-iterations"));
        } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            std::printf("usage: %s [--operation both|read|read-write] [--warmup N] "
                        "[--repeat N] [--min-ms N] [--profile-iterations N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") + argv[i]);
        }
    }
    if (options.warmup < 0 || options.repeat <= 0 || options.min_time_ms <= 0 ||
        options.profile_iterations < 0) {
        throw std::invalid_argument("invalid non-positive benchmark count");
    }
    if (options.profile_iterations > 0 && options.operation == Operation::Both) {
        throw std::invalid_argument("profiling requires one explicit operation");
    }
    return options;
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
    bytes[0] = static_cast<std::uint8_t>(value & 0xffU);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

std::vector<std::uint8_t> make_q8(std::int32_t rows, std::int32_t columns,
                                  std::uint32_t seed) {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * (columns / 32) *
                                    kQ8BlockBytes);
    std::uint32_t state = seed;
    for (std::size_t offset = 0; offset < bytes.size(); offset += kQ8BlockBytes) {
        write_u16(bytes.data() + offset, 0x1800U); // exact FP16 scale 2^-9
        for (int item = 0; item < 32; ++item) {
            state = state * 1664525U + 1013904223U;
            const int value = static_cast<int>((state >> 24U) % 63U) - 31;
            bytes[offset + 2 + item] = static_cast<std::uint8_t>(value);
        }
    }
    return bytes;
}

ninfer::DeviceBuffer copy_bytes(const void* source, std::size_t bytes) {
    ninfer::DeviceBuffer result(bytes);
    result.copy_from_host(source, bytes);
    return result;
}

ninfer::Weight q8_weight(const void* data, std::uint64_t bytes, std::int32_t rows,
                         std::int32_t columns) {
    ninfer::Weight weight{};
    weight.payload = data;
    weight.payload_bytes = bytes;
    weight.qdata = data;
    weight.qtype = ninfer::QType::GGML_Q8_0;
    weight.group_size = 32;
    weight.group = 32;
    weight.layout = ninfer::QuantLayout::GgmlBlockRow;
    weight.ndim = 2;
    weight.n = rows;
    weight.k = columns;
    weight.shape[0] = rows;
    weight.shape[1] = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    return weight;
}

struct Fixture {
    ninfer::DeviceBuffer residual;
    ninfer::DeviceBuffer norm_weight;
    ninfer::DeviceBuffer down_data;
    ninfer::DeviceBuffer up_data;
    ninfer::DeviceBuffer write_weight;
    ninfer::DeviceBuffer x;
    ninfer::DeviceBuffer write_scale;
    ninfer::WorkspaceArena workspace;
    ninfer::Tensor residual_tensor;
    ninfer::Tensor norm_tensor;
    ninfer::Weight down;
    ninfer::Weight up;
    ninfer::Tensor write_tensor;
    ninfer::Tensor x_tensor;
    ninfer::Tensor scale_tensor;

    Fixture()
        : residual(ninfer::bench::make_bf16(kFlat)),
          norm_weight(make_f32(kFlat, 0x7101U)),
          down_data(make_q8_device(kRank, kFlat, 0x7102U)),
          up_data(make_q8_device(kFlat, kRank, 0x7103U)),
          write_weight(make_f32(static_cast<std::size_t>(kBranches) * kFlat, 0x7104U, 0.003F)),
          x(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t)),
          write_scale(static_cast<std::size_t>(kBranches) * sizeof(std::uint16_t)),
          workspace(ninfer::ops::gated_residual_workspace_capacity_bytes()),
          residual_tensor(residual.p, ninfer::DType::BF16, {kHidden, kBranches}),
          norm_tensor(norm_weight.p, ninfer::DType::FP32, {kFlat}),
          down(q8_weight(down_data.p, down_data.bytes, kRank, kFlat)),
          up(q8_weight(up_data.p, up_data.bytes, kFlat, kRank)),
          write_tensor(write_weight.p, ninfer::DType::FP32, {kFlat, kBranches}),
          x_tensor(x.p, ninfer::DType::BF16, {kHidden}),
          scale_tensor(write_scale.p, ninfer::DType::BF16, {kBranches}) {}

private:
    static ninfer::DeviceBuffer make_f32(std::size_t count, std::uint32_t seed,
                                         float magnitude = 0.2F) {
        std::vector<float> values(count);
        std::uint32_t state = seed;
        for (std::size_t i = 0; i < count; ++i) {
            state = state * 1664525U + 1013904223U;
            const float unit = static_cast<float>((state >> 8U) & 0x00ffffffU) /
                               static_cast<float>(0x01000000U);
            values[i] = (2.0F * unit - 1.0F) * magnitude;
        }
        return copy_bytes(values.data(), values.size() * sizeof(float));
    }

    static ninfer::DeviceBuffer make_q8_device(std::int32_t rows, std::int32_t columns,
                                                std::uint32_t seed) {
        const std::vector<std::uint8_t> bytes = make_q8(rows, columns, seed);
        return copy_bytes(bytes.data(), bytes.size());
    }
};

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        const Options options = parse_args(argc, argv);
        Fixture fixture;
        cudaStream_t stream = nullptr;
        const auto read = [&](cudaStream_t target_stream) {
            ninfer::ops::gated_residual_read(fixture.residual_tensor, fixture.norm_tensor,
                                             fixture.down, fixture.up, fixture.x_tensor,
                                             fixture.workspace, target_stream);
        };
        const auto read_write = [&](cudaStream_t target_stream) {
            ninfer::ops::gated_residual_read_write(
                fixture.residual_tensor, fixture.norm_tensor, fixture.down, fixture.up,
                fixture.write_tensor, fixture.x_tensor, fixture.scale_tensor, fixture.workspace,
                target_stream);
        };

        ninfer::bench::print_device_caps("gated_residual");
        std::printf("geometry: C=1 H=2560 branches=4 flat=10240 rank=320; Q8_0 down/up, "
                    "FP32 norm/write\n");
        if (options.operation == Operation::Both || options.operation == Operation::Read) {
            const auto result = ninfer::bench::bench_loop(
                read, kReadInterfaceBytes, options.warmup, options.repeat, options.min_time_ms);
            ninfer::bench::print_result("gated_residual_read", result);
        }
        if (options.operation == Operation::Both || options.operation == Operation::ReadWrite) {
            const auto result = ninfer::bench::bench_loop(read_write, kReadWriteInterfaceBytes,
                                                          options.warmup, options.repeat,
                                                          options.min_time_ms);
            ninfer::bench::print_result("gated_residual_read_write", result);
        }

        if (options.profile_iterations > 0) {
            for (int i = 0; i < options.warmup; ++i) {
                if (options.operation == Operation::Read) {
                    read(stream);
                } else {
                    read_write(stream);
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStart());
            for (int i = 0; i < options.profile_iterations; ++i) {
                if (options.operation == Operation::Read) {
                    read(stream);
                } else {
                    read_write(stream);
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
}
