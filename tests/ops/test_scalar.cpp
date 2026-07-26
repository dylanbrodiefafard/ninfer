// Bit-exact public-contract qualification for the finite typed scalar state
// transitions.  Expected values are computed directly from the logical input
// state; no launcher or kernel implementation is used as an oracle.
#include "ninfer/ops/scalar.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

template <typename T>
void store(GuardedDeviceBuffer& buffer, T value) {
    buffer.copy_from_host(&value, sizeof(value));
}

template <typename T>
T load(const GuardedDeviceBuffer& buffer) {
    T value{};
    buffer.copy_to_host(&value, sizeof(value));
    return value;
}

int set_and_increment_i32_contract(cudaStream_t stream) {
    GuardedDeviceBuffer state(sizeof(std::int32_t));
    store<std::int32_t>(state, -918273);
    Tensor scalar(state.data(), DType::I32, {1});

    constexpr std::int32_t assigned = 123456789;
    ops::set_i32_scalar(scalar, assigned, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("set_i32_scalar transition",
                             std::vector<std::int32_t>{load<std::int32_t>(state)},
                             std::vector<std::int32_t>{assigned});
    failures += state.verify_guards("set_i32_scalar destination");

    const std::int32_t expected_increment = assigned + 1;
    ops::increment_i32_scalar(scalar, stream);
    cuda_synchronize(stream);
    failures += verify_exact("increment_i32_scalar transition",
                             std::vector<std::int32_t>{load<std::int32_t>(state)},
                             std::vector<std::int32_t>{expected_increment});
    failures += state.verify_guards("increment_i32_scalar destination");
    return failures;
}

int assign_i32_contract(cudaStream_t stream) {
    GuardedDeviceBuffer source(sizeof(std::int32_t));
    GuardedDeviceBuffer destination(sizeof(std::int32_t));
    constexpr std::int32_t source_value      = -19088743;
    constexpr std::int32_t destination_value = 1985229328;
    store<std::int32_t>(source, source_value);
    store<std::int32_t>(destination, destination_value);
    Tensor source_tensor(source.data(), DType::I32, {1});
    Tensor destination_tensor(destination.data(), DType::I32, {1});

    ops::assign_i32_scalar(source_tensor, destination_tensor, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("assign_i32_scalar destination transition",
                             std::vector<std::int32_t>{load<std::int32_t>(destination)},
                             std::vector<std::int32_t>{source_value});
    failures += verify_exact("assign_i32_scalar source remains unchanged",
                             std::vector<std::int32_t>{load<std::int32_t>(source)},
                             std::vector<std::int32_t>{source_value});
    failures += source.verify_guards("assign_i32_scalar source");
    failures += destination.verify_guards("assign_i32_scalar destination");
    return failures;
}

int increment_i64_contract(cudaStream_t stream) {
    GuardedDeviceBuffer state(sizeof(std::int64_t));
    constexpr std::int64_t initial = (std::int64_t{1} << 45) + 987654321;
    store<std::int64_t>(state, initial);
    Tensor scalar(state.data(), DType::I64, {1});

    ops::increment_i64_scalar(scalar, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("increment_i64_scalar transition",
                             std::vector<std::int64_t>{load<std::int64_t>(state)},
                             std::vector<std::int64_t>{initial + 1});
    failures += state.verify_guards("increment_i64_scalar destination");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    int failures = 0;
    failures += set_and_increment_i32_contract(stream);
    failures += assign_i32_contract(stream);
    failures += increment_i64_contract(stream);

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " scalar public contract\n";
    return failures == 0 ? 0 : 1;
}
