#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;

Json normative_directory() {
    return {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"name", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "FP32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"name", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"name", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"name", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "Q5G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"name", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "Q6G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"name", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "W8G32_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                        {{"name", "ggml_q8"},
                         {"kind", "tensor"},
                         {"shape", {1, 32}},
                         {"format", "Q8_0"},
                         {"layout", "ggml-block-row-v1"},
                         {"offset", 3584},
                         {"bytes", 34}},
                        {{"name", "ggml_q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 256}},
                         {"format", "Q4_K"},
                         {"layout", "ggml-block-row-v1"},
                         {"offset", 3840},
                         {"bytes", 144}},
                        {{"name", "ggml_q5"},
                         {"kind", "tensor"},
                         {"shape", {1, 256}},
                         {"format", "Q5_K"},
                         {"layout", "ggml-block-row-v1"},
                         {"offset", 4096},
                         {"bytes", 176}},
                        {{"name", "ggml_q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 256}},
                         {"format", "Q6_K"},
                         {"layout", "ggml-block-row-v1"},
                         {"offset", 4352},
                         {"bytes", 210}},
                        {{"name", "ggml_iq1"},
                         {"kind", "tensor"},
                         {"shape", {2, 1, 256}},
                         {"format", "IQ1_S"},
                         {"layout", "ggml-block-row-v1"},
                         {"offset", 4608},
                         {"bytes", 100}},
                        {{"name", "ggml_iq2"},
                         {"kind", "tensor"},
                         {"shape", {1, 256}},
                         {"format", "IQ2_XXS"},
                         {"layout", "ggml-block-row-v1"},
                         {"offset", 4864},
                         {"bytes", 66}},
                        {{"name", "ggml_iq4"},
                         {"kind", "tensor"},
                         {"shape", {1, 32}},
                         {"format", "IQ4_NL"},
                         {"layout", "ggml-block-row-v1"},
                         {"offset", 5120},
                         {"bytes", 18}},
                    })},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows   = StorageLayout::RowSplitK128V1;
    constexpr StorageLayout ggml   = StorageLayout::GgmlBlockRowV1;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};
    const std::array<std::uint64_t, 2> ggml32_shape  = {1, 32};
    const std::array<std::uint64_t, 2> ggml256_shape = {1, 256};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264 ||
        tensor_encoded_size(ggml, NumericFormat::Q8_0, ggml32_shape) != 34 ||
        tensor_encoded_size(ggml, NumericFormat::Q4_K, ggml256_shape) != 144 ||
        tensor_encoded_size(ggml, NumericFormat::Q5_K, ggml256_shape) != 176 ||
        tensor_encoded_size(ggml, NumericFormat::Q6_K, ggml256_shape) != 210 ||
        tensor_encoded_size(ggml, NumericFormat::IQ1_S, ggml256_shape) != 50 ||
        tensor_encoded_size(ggml, NumericFormat::IQ2_XXS, ggml256_shape) != 66 ||
        tensor_encoded_size(ggml, NumericFormat::IQ4_NL, ggml32_shape) != 18) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
}

void test_fp8_row_layout() {
    const std::array<std::uint64_t, 2> shape{2, 4};
    const auto geometry = ninfer::artifact::row_scale_geometry(
        NumericFormat::FP8_E4M3FN_ROW_BF16S, shape);
    if (geometry.code_plane_bytes != 8 || geometry.scale_plane_offset != 256 ||
        geometry.scale_plane_bytes != 4 || geometry.encoded_bytes != 260) {
        throw std::runtime_error("FP8 row-scale geometry differs from exact byte layout");
    }
    auto directory = normative_directory();
    directory["objects"].push_back({{"name", "fp8"}, {"kind", "tensor"}, {"shape", {2, 4}},
        {"format", "FP8_E4M3FN_ROW_BF16S"}, {"layout", "row-scale-v1"},
        {"offset", 5376}, {"bytes", 260}});
    auto fixture = write_fixture(directory, "fp8-row-scale");
    Reader reader(fixture.path);
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find("fp8"));
    if (tensor == nullptr || tensor->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        tensor->layout != StorageLayout::RowScaleV1 || reader.payload(*tensor).data.size() != 260) {
        throw std::runtime_error("FP8 row-scale directory/payload did not round trip");
    }
    expect_artifact_error([&] {
        (void)ninfer::artifact::tensor_encoded_size(StorageLayout::RowSplitK128V1,
            NumericFormat::FP8_E4M3FN_ROW_BF16S, shape);
    }, "FP8 cannot use grouped integer layout");
}

void test_expert_nvfp4_layout() {
    for (const auto shape : {std::array<std::uint64_t, 3>{512, 640, 2560},
                              std::array<std::uint64_t, 3>{512, 2560, 640}}) {
        const auto g = ninfer::artifact::expert_block_scale_geometry(
            NumericFormat::NVFP4_EXPERT_F32M, shape);
        if (g.code_plane_bytes != 419430400 || g.scale_plane_offset != 419430400 ||
            g.scale_plane_bytes != 52428800 || g.weight_multiplier_offset != 471859200 ||
            g.input_multiplier_offset != 471861248 || g.encoded_bytes != 471863296) {
            throw std::runtime_error("source expert bank geometry mismatch");
        }
        if (ninfer::artifact::tensor_encoded_size(StorageLayout::ExpertBlockScaleK16M128x4V1,
                NumericFormat::NVFP4_EXPERT_F32M, shape) != g.encoded_bytes) {
            throw std::runtime_error("expert layout dispatch mismatch");
        }
    }
    const std::array<std::uint64_t, 2> matrix{640, 2560};
    expect_artifact_error([&] { ninfer::artifact::expert_block_scale_geometry(
        NumericFormat::NVFP4_EXPERT_F32M, matrix); }, "rank-two expert bank");
    auto directory = normative_directory();
    directory["objects"].push_back({{"name", "experts"}, {"kind", "tensor"},
        {"shape", {2, 128, 64}}, {"format", "NVFP4_EXPERT_F32M"},
        {"layout", "expert-blockscale-k16-m128x4-v1"}, {"offset", 5376}, {"bytes", 9232}});
    auto fixture = write_fixture(directory, "expert-nvfp4");
    Reader reader(fixture.path);
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find("experts"));
    if (tensor == nullptr || tensor->format != NumericFormat::NVFP4_EXPERT_F32M ||
        tensor->layout != StorageLayout::ExpertBlockScaleK16M128x4V1 ||
        reader.payload(*tensor).data.size() != 9232) {
        throw std::runtime_error("NVFP4 expert directory/payload did not round trip");
    }
}

void test_partition_nvfp4_layout() {
    for (const auto shape : {std::array<std::uint64_t, 3>{3, 4, 160},
                             std::array<std::uint64_t, 3>{128, 2500012, 160}}) {
        const auto g = ninfer::artifact::partition_block_scale_geometry(
            NumericFormat::NVFP4_PARTITION_F32M, shape);
        if (g.row_bytes != 90 || g.multiplier_offset != shape[0] * shape[1] * 90 ||
            g.encoded_bytes != shape[0] * shape[1] * 90 + 4 * shape[0]) {
            throw std::runtime_error("partition NVFP4 geometry mismatch");
        }
    }
    auto directory = normative_directory();
    directory["objects"].push_back({{"name", "ple"}, {"kind", "tensor"},
        {"shape", {3, 4, 160}}, {"format", "NVFP4_PARTITION_F32M"},
        {"layout", "partitioned-row-blockscale-k16-v1"}, {"offset", 5376}, {"bytes", 1092}});
    auto fixture = write_fixture(directory, "partition-nvfp4");
    Reader reader(fixture.path);
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find("ple"));
    if (!tensor || tensor->format != NumericFormat::NVFP4_PARTITION_F32M ||
        tensor->layout != StorageLayout::PartitionedRowBlockScaleK16V1 ||
        reader.payload(*tensor).data.size() != 1092) {
        throw std::runtime_error("partition NVFP4 directory/payload did not round trip");
    }
}

void test_tensor_fp8_layout() {
    const std::array<std::uint64_t, 2> shape{16, 160};
    const auto g = ninfer::artifact::tensor_scale_geometry(
        NumericFormat::FP8_E4M3FN_TENSOR_BF16S, shape);
    if (g.code_plane_bytes != 2560 || g.scale_offset != 2560 || g.encoded_bytes != 2562) {
        throw std::runtime_error("tensor FP8 geometry mismatch");
    }
    auto directory = normative_directory();
    directory["objects"].push_back({{"name", "ple"}, {"kind", "tensor"},
        {"shape", {16, 160}}, {"format", "FP8_E4M3FN_TENSOR_BF16S"},
        {"layout", "tensor-scale-v1"}, {"offset", 5376}, {"bytes", 2562}});
    auto fixture = write_fixture(directory, "tensor-fp8");
    Reader reader(fixture.path);
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find("ple"));
    if (tensor == nullptr || tensor->format != NumericFormat::FP8_E4M3FN_TENSOR_BF16S ||
        tensor->layout != StorageLayout::TensorScaleV1 || reader.payload(*tensor).data.size() != 2562) {
        throw std::runtime_error("tensor FP8 directory/payload did not round trip");
    }
}

void test_calibrated_fp8_layout() {
    const std::array<std::uint64_t, 2> shape{3, 3};
    const auto g = ninfer::artifact::tensor_calibrated_geometry(
        NumericFormat::FP8_E4M3FN_TENSOR_F32M, shape);
    if (g.multiplier_offset != 12 || g.encoded_bytes != 20) {
        throw std::runtime_error("calibrated FP8 aligned geometry mismatch");
    }
    auto directory = normative_directory();
    directory["objects"].push_back({{"name", "projection"}, {"kind", "tensor"},
        {"shape", {3, 3}}, {"format", "FP8_E4M3FN_TENSOR_F32M"},
        {"layout", "tensor-calibrated-v1"}, {"offset", 5376}, {"bytes", 20}});
    auto fixture = write_fixture(directory, "calibrated-fp8");
    Reader reader(fixture.path);
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find("projection"));
    if (!tensor || tensor->format != NumericFormat::FP8_E4M3FN_TENSOR_F32M ||
        tensor->layout != StorageLayout::TensorCalibratedV1 ||
        reader.payload(*tensor).data.size() != 20) {
        throw std::runtime_error("calibrated FP8 directory/payload did not round trip");
    }
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 15 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 15> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4",       "q5",       "q6",
        "w8",       "ggml_q8", "ggml_q4",   "ggml_q5", "ggml_q6", "ggml_iq1",
        "ggml_iq2", "ggml_iq4",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5       = std::get_if<TensorDescriptor>(reader.find("q5"));
    const auto* iq1      = std::get_if<TensorDescriptor>(reader.find("ggml_iq1"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
    if (iq1 == nullptr || iq1->shape != std::vector<std::uint64_t>({2, 1, 256}) ||
        iq1->format != NumericFormat::IQ1_S ||
        iq1->layout != StorageLayout::GgmlBlockRowV1) {
        throw std::runtime_error("GGML fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture                     = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][9]["bytes"] = 143;
        auto fixture = write_fixture(directory, "wrong_ggml_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong GGML encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][10]["shape"] = {1, 255};
        directory["objects"][10]["bytes"] = 176;
        auto fixture = write_fixture(directory, "partial_ggml_block");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "partial GGML K block");
    }
    {
        auto directory                      = normative_directory();
        directory["objects"][11]["format"] = "BF16";
        auto fixture = write_fixture(directory, "wrong_ggml_format_layout");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong GGML format/layout");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture                      = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
    {
        auto directory = normative_directory();
        auto fixture =
            write_fixture(directory, "legacy_v1", ninfer::test::artifact_fixture::kV1Magic);
        try {
            Reader reader(fixture.path);
        } catch (const ninfer::artifact::ArtifactError& error) {
            if (std::string_view(error.what())
                    .find("python3 -m tools.artifact.migrate_v1_to_v2 <artifact>") ==
                std::string_view::npos) {
                throw std::runtime_error("v1 rejection omitted the migration command");
            }
            return;
        }
        throw std::runtime_error("v1 artifact was accepted");
    }
}

void test_file_generation_identity() {
    auto original = write_fixture(normative_directory(), "file-identity-original");
    auto replacement = write_fixture(normative_directory(), "file-identity-replacement");
    const auto initial = Reader(original.path).file_identity();
    if (initial.empty() || Reader(original.path).file_identity() != initial) {
        throw std::runtime_error("unchanged artifact file identity is not stable");
    }
    if (Reader(replacement.path).file_identity() == initial) {
        throw std::runtime_error("distinct files with the same artifact identity alias");
    }
    // Same pathname, same byte length, same public identity; a new inode must
    // invalidate persistent state rather than reusing the replaced model's KV.
    std::filesystem::rename(replacement.path, original.path);
    const auto replaced = Reader(original.path).file_identity();
    if (replaced == initial) { throw std::runtime_error("artifact replacement was not detected"); }
    const auto before = std::filesystem::last_write_time(original.path);
    {
        const auto payload_offset = Reader(original.path).payload_offset();
        std::fstream file(original.path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(payload_offset));
        file.put('Z');
        if (!file) { throw std::runtime_error("failed to modify artifact fixture payload"); }
    }
    std::filesystem::last_write_time(original.path, before - std::chrono::seconds(2));
    if (Reader(original.path).file_identity() == replaced) {
        throw std::runtime_error("artifact modification was not detected");
    }
}

std::uint64_t locked_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.starts_with("VmLck:")) { return std::stoull(line.substr(6)) * 1024; }
    }
    throw std::runtime_error("missing Linux locked-memory accounting");
}

void test_resident_payload() {
    const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    auto directory = normative_directory();
    auto tensor = directory["objects"][1];
    tensor["shape"] = {page + 17};
    tensor["bytes"] = 2 * (page + 17);
    directory["objects"] = Json::array({tensor});
    auto fixture = write_fixture(directory, "resident-payload");
    const auto locked_span = 3 * page; // Unaligned payload spans three host pages.
    const auto before = locked_bytes();
    ninfer::artifact::ResidentPayload retained;
    std::vector<std::byte> expected;
    {
        Reader reader(fixture.path);
        const auto& descriptor = *reader.find("bf16");
        const auto source = reader.payload(descriptor).data;
        expected.assign(source.begin(), source.end());
        retained = reader.resident_payload(descriptor);
        if (locked_bytes() != before + locked_span || retained.locked_bytes != locked_span ||
            retained.data.size() != 2 * (page + 17)) {
            throw std::runtime_error("resident tensor did not lock its exact page span");
        }
        // Independently owned model instances must not unlock each other's pages.
        {
            auto second = reader.resident_payload(descriptor);
            if (locked_bytes() != before + 2 * locked_span) {
                throw std::runtime_error("resident mappings unexpectedly share lock lifetime");
            }
        }
        if (locked_bytes() != before + locked_span) {
            throw std::runtime_error("destroying another owner released live tensor residency");
        }
    }
    const auto base = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(retained.data.data()) / page * page);
    std::array<unsigned char, 3> resident{};
    if (::mincore(base, locked_span, resident.data()) != 0 ||
        !std::all_of(resident.begin(), resident.end(), [](unsigned char value) { return (value & 1) != 0; }) ||
        !std::equal(retained.data.begin(), retained.data.end(), expected.begin(), expected.end())) {
        throw std::runtime_error("resident tensor lost bytes or residency after Reader destruction");
    }
    auto moved = std::move(retained);
    moved.backing.reset();
    if (locked_bytes() != before) { throw std::runtime_error("resident mapping leaked locked pages"); }

    // Isolate limit changes from this test process and the rest of CTest. Drop
    // root privileges in the child so CAP_IPC_LOCK cannot bypass a zero limit.
    const auto child = ::fork();
    if (child < 0) { throw std::runtime_error("fork failed for memory-lock failure test"); }
    if (child == 0) {
        try {
            Reader reader(fixture.path);
            auto first = reader.resident_payload(*reader.find("bf16"));
            const auto first_locked = locked_bytes();
            if (::geteuid() == 0 && ::setuid(65534) != 0) { ::_exit(2); }
            struct rlimit limit {0, 0};
            if (::setrlimit(RLIMIT_MEMLOCK, &limit) != 0) { ::_exit(3); }
            for (int i = 0; i < 16; ++i) {
                try {
                    (void)reader.resident_payload(*reader.find("bf16"));
                    ::_exit(4);
                } catch (const ninfer::artifact::ArtifactError& error) {
                    if (std::string_view(error.what()).find("disk-backed fallback is disabled") ==
                        std::string_view::npos) { ::_exit(5); }
                }
            }
            if (locked_bytes() != first_locked) { ::_exit(6); }
            first.backing.reset();
            if (locked_bytes() != 0) { ::_exit(7); }
            ::_exit(0);
        } catch (...) { ::_exit(8); }
    }
    int status = 0;
    if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("resident payload failure/cleanup test failed: " + std::to_string(status));
    }
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_expert_nvfp4_layout();
        test_tensor_fp8_layout();
        test_calibrated_fp8_layout();
        test_partition_nvfp4_layout();
        test_fp8_row_layout();
        test_normative_fixture();
        test_common_validation();
        test_file_generation_identity();
        test_resident_payload();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
