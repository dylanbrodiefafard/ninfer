#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                        })},
        },
        "materialization");
}

ninfer::test::artifact_fixture::TemporaryArtifact write_parallel_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    Json objects = Json::array();
    for (std::uint64_t i = 0; i < 9; ++i) {
        objects.push_back({
            {"name", "weights/parallel-" + std::to_string(i)},
            {"kind", "tensor"},
            {"shape", {1}},
            {"format", "BF16"},
            {"layout", "contiguous-le-v1"},
            {"offset", i * 8192},
            {"bytes", 2},
        });
    }
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", std::move(objects)},
        },
        "parallel-materialization");
}

ninfer::test::artifact_fixture::TemporaryArtifact write_ggml_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "ggml-weights"}}},
            {"objects",
             Json::array({
                 {{"name", "experts/iq1"},
                  {"kind", "tensor"},
                  {"shape", {2, 1, 256}},
                  {"format", "IQ1_S"},
                  {"layout", "ggml-block-row-v1"},
                  {"offset", 0},
                  {"bytes", 100}},
             })},
        },
        "ggml-materialization");
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Function>
void require_artifact_error(Function&& function, const char* message) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) {
        return;
    }
    throw std::runtime_error(message);
}

template <class Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.map_tensor_on_host(retained_tensor);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 3 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.mapped_tensor_objects.size() == 1 &&
                    validation_plan.device_objects.empty() &&
                    validation_plan.device_capacity_bytes == 0 &&
                    validation_plan.mapped_tensor_objects.front().bytes == kSecondTensor.size(),
                "mapped and validate-only tensors were not distinguished in the plan");

        ninfer::artifact::Binder rejection_binder(reader);
        const auto rejected_resource = rejection_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        require_artifact_error(
            [&] { rejection_binder.map_tensor_on_host(rejected_resource); },
            "resource was accepted as a mapped host tensor");
        rejection_binder.retain_on_host(rejected_resource);
        const auto rejected_tensor = rejection_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        require_artifact_error(
            [&] { rejection_binder.retain_on_host(rejected_tensor); },
            "tensor was accepted as a copied host resource");
        rejection_binder.map_tensor_on_host(rejected_tensor);
        require_artifact_error(
            [&] { rejection_binder.materialize_on_device(rejected_tensor); },
            "mapped tensor accepted a second placement");
        const auto rejected_second = rejection_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        rejection_binder.validate_only(rejected_second);
        const auto rejection_plan = rejection_binder.finish();
        require(rejection_plan.host_objects.size() == 1 &&
                    rejection_plan.mapped_tensor_objects.size() == 1 &&
                    rejection_plan.device_objects.empty(),
                "failed placement attempts changed the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cerr << "FAIL: no usable CUDA device\n";
            return 1;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cerr << "FAIL: no usable CUDA device\n";
            return 1;
        }

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.map_tensor_on_host(tensor);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 3 && plan.host_objects.size() == 1 &&
                    plan.mapped_tensor_objects.size() == 1 && plan.device_objects.size() == 1 &&
                    plan.device_capacity_bytes == kSecondTensor.size(),
                "binder produced the wrong materialization plan");

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_data(second),
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");
        const auto mapped = materialized.mapped_tensor_bytes(tensor);
        require(std::equal(mapped.begin(), mapped.end(), kTensor.begin(), kTensor.end()),
                "mapped tensor payload differs from the artifact");
        require_artifact_error([&] { (void)materialized.device_data(tensor); },
                               "mapped tensor exposed a device address");
        require_artifact_error([&] { (void)materialized.resource_bytes(tensor); },
                               "mapped tensor exposed copied resource bytes");
        require_artifact_error([&] { (void)materialized.mapped_tensor_bytes(resource); },
                               "copied resource exposed mapped tensor bytes");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 1 && stats.mapped_tensor_count == 1 &&
                    stats.resource_count == 1 && stats.h2d_bytes == kSecondTensor.size() &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.mapped_tensor_bytes == kTensor.size() &&
                    stats.file_bytes == kResource.size() + kSecondTensor.size() &&
                    stats.peak_staging_bytes ==
                        ninfer::artifact::Reader::direct_io_alignment,
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");

        ninfer::artifact::MaterializedArtifact after_reader;
        ninfer::artifact::ObjectHandle lifetime_tensor;
        {
            ninfer::artifact::Reader lifetime_reader(fixture.path);
            ninfer::artifact::Binder lifetime_binder(lifetime_reader);
            const auto lifetime_resource = lifetime_binder.require_resource(
                "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
            lifetime_binder.validate_only(lifetime_resource);
            lifetime_tensor = ninfer::artifact::bind_tensor(
                lifetime_binder, "weights/test", ninfer::artifact::NumericFormat::BF16, {2},
                ninfer::artifact::TensorPlacement::MappedHost);
            const auto lifetime_second = lifetime_binder.require_tensor(
                "weights/second", ninfer::artifact::NumericFormat::BF16,
                ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
            lifetime_binder.validate_only(lifetime_second);
            const auto lifetime_plan = lifetime_binder.finish();
            after_reader =
                ninfer::artifact::materialize(lifetime_reader, lifetime_plan, device);
        }
        const auto mapped_after_reader = after_reader.mapped_tensor_bytes(lifetime_tensor);
        require(std::equal(mapped_after_reader.begin(), mapped_after_reader.end(), kTensor.begin(),
                           kTensor.end()),
                "mapped tensor did not retain the Reader mapping lifetime");
        const auto& lifetime_stats = after_reader.stats();
        require(lifetime_stats.tensor_count == 0 && lifetime_stats.mapped_tensor_count == 1 &&
                    lifetime_stats.mapped_tensor_bytes == kTensor.size() &&
                    lifetime_stats.resource_count == 0 && lifetime_stats.file_bytes == 0 &&
                    lifetime_stats.h2d_bytes == 0 &&
                    lifetime_stats.device_capacity_bytes == 0 &&
                    lifetime_stats.retained_resource_bytes == 0 &&
                    lifetime_stats.peak_staging_bytes == 0 &&
                    lifetime_stats.upload_seconds == 0.0,
                "mapped-only materialization statistics are not exact");
        require_artifact_error([&] { (void)after_reader.device_arena(); },
                               "mapped-only artifact exposed a device arena");

        auto parallel_fixture = write_parallel_fixture();
        ninfer::artifact::Reader parallel_reader(parallel_fixture.path);
        ninfer::artifact::Binder parallel_binder(parallel_reader);
        std::vector<ninfer::artifact::ObjectHandle> parallel_handles;
        constexpr std::array<std::uint64_t, 1> parallel_shape = {1};
        for (std::size_t i = 0; i < 9; ++i) {
            const auto handle = parallel_binder.require_tensor(
                "weights/parallel-" + std::to_string(i),
                ninfer::artifact::NumericFormat::BF16,
                ninfer::artifact::StorageLayout::ContiguousLeV1, parallel_shape);
            parallel_binder.materialize_on_device(handle);
            parallel_handles.push_back(handle);
        }
        const auto parallel_plan = parallel_binder.finish();
        auto parallel = ninfer::artifact::materialize(parallel_reader, parallel_plan, device);
        for (std::size_t i = 0; i < parallel_handles.size(); ++i) {
            std::array<std::byte, 2> value{};
            CUDA_CHECK(cudaMemcpy(value.data(), parallel.device_data(parallel_handles[i]),
                                  value.size(), cudaMemcpyDeviceToHost));
            const std::byte expected{static_cast<unsigned char>(i + 1)};
            require(value[0] == expected && value[1] == expected,
                    "parallel materialization tensor payload differs from the artifact");
        }
        require(parallel.stats().tensor_count == parallel_handles.size() &&
                    parallel.stats().h2d_bytes == parallel_handles.size() * 2 &&
                    parallel.stats().peak_staging_bytes ==
                        8 * ninfer::artifact::Reader::direct_io_alignment,
                "parallel materialization did not exercise eight reusable staging slots");

        auto ggml_fixture = write_ggml_fixture();
        ninfer::artifact::Reader ggml_reader(ggml_fixture.path);
        ninfer::artifact::Binder ggml_binder(ggml_reader);
        const auto ggml_handle = ninfer::artifact::bind_tensor(
            ggml_binder, "experts/iq1", ninfer::artifact::NumericFormat::IQ1_S, {2, 1, 256},
            ninfer::artifact::TensorPlacement::Device);
        const auto ggml_plan = ggml_binder.finish();
        auto ggml_materialized = ninfer::artifact::materialize(ggml_reader, ggml_plan, device);
        const auto bank = ninfer::artifact::materialized_ggml_block_weight(
            ggml_materialized, ggml_handle, ninfer::artifact::NumericFormat::IQ1_S, {2, 1, 256});
        const auto expert = ninfer::artifact::ggml_block_matrix_view(bank, 1);
        require(bank.ndim == 3 && bank.shape[0] == 2 && bank.shape[1] == 1 &&
                    bank.shape[2] == 256 && bank.payload_bytes == 100 &&
                    expert.ndim == 2 && expert.shape[0] == 1 && expert.shape[1] == 256 &&
                    expert.n == 1 && expert.k == 256 && expert.payload_bytes == 50 &&
                    static_cast<const std::byte*>(expert.payload) ==
                        static_cast<const std::byte*>(bank.payload) + 50 &&
                    expert.qdata == expert.payload,
                "GGML expert slice did not preserve exact rank-two byte geometry");
        std::array<std::byte, 50> expert_bytes{};
        CUDA_CHECK(cudaMemcpy(expert_bytes.data(), expert.payload, expert_bytes.size(),
                              cudaMemcpyDeviceToHost));
        require(std::all_of(expert_bytes.begin(), expert_bytes.end(),
                            [](std::byte value) { return value == std::byte{1}; }),
                "GGML expert slice changed source block bytes");
        require_invalid_argument(
            [&] {
                (void)ninfer::artifact::materialized_ggml_block_weight(
                    ggml_materialized, ggml_handle, ninfer::artifact::NumericFormat::NVFP4,
                    {2, 1, 256});
            },
            "non-GGML format was accepted by GGML typed binding");
        auto wrong_qtype = bank;
        wrong_qtype.qtype = ninfer::QType::NVFP4;
        require_invalid_argument(
            [&] { (void)ninfer::artifact::ggml_block_matrix_view(wrong_qtype, 0); },
            "non-GGML qtype was accepted by GGML expert view");
        try {
            (void)ninfer::artifact::ggml_block_matrix_view(bank, 2);
            throw std::runtime_error("out-of-range GGML expert index was accepted");
        } catch (const std::invalid_argument&) {
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
