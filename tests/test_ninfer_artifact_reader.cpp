#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
}

void test_fp8_row_layout() {
    const std::array<std::uint64_t, 2> shape{2, 4};
    const auto geometry =
        ninfer::artifact::row_scale_geometry(NumericFormat::FP8_E4M3FN_ROW_BF16S, shape);
    if (geometry.code_plane_bytes != 8 || geometry.scale_plane_offset != 256 ||
        geometry.scale_plane_bytes != 4 || geometry.encoded_bytes != 260) {
        throw std::runtime_error("FP8 row-scale geometry differs from exact byte layout");
    }
    auto directory = normative_directory();
    directory["objects"].push_back({{"name", "fp8"},
                                    {"kind", "tensor"},
                                    {"shape", {2, 4}},
                                    {"format", "FP8_E4M3FN_ROW_BF16S"},
                                    {"layout", "row-scale-v1"},
                                    {"offset", 3584},
                                    {"bytes", 260}});
    auto fixture = write_fixture(directory, "fp8-row-scale");
    Reader reader(fixture.path);
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find("fp8"));
    if (tensor == nullptr || tensor->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        tensor->layout != StorageLayout::RowScaleV1 || reader.payload(*tensor).data.size() != 260) {
        throw std::runtime_error("FP8 row-scale directory/payload did not round trip");
    }
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::tensor_encoded_size(StorageLayout::RowSplitK128V1,
                                                        NumericFormat::FP8_E4M3FN_ROW_BF16S, shape);
        },
        "FP8 cannot use grouped integer layout");
}

void test_expert_tiles_layout() {
    // Qwen3.8-Flash-Next records: 800 gate|up and 400 down chunks of 1664 (F3) or 2304 bytes.
    const std::array<std::uint64_t, 3> flash_next{3, 2560, 640};
    const auto f3 =
        ninfer::artifact::expert_tiles_geometry(NumericFormat::F3G32_E4M3FN, flash_next);
    const auto e2m1 =
        ninfer::artifact::expert_tiles_geometry(NumericFormat::E2M1G16_E4M3FN, flash_next);
    if (f3.chunk_bytes != 1664 || f3.gate_up_chunks != 800 || f3.down_chunks != 400 ||
        f3.record_bytes != 1996800 || f3.encoded_bytes != 3 * 1996800 || e2m1.chunk_bytes != 2304 ||
        e2m1.record_bytes != 2764800) {
        throw std::runtime_error("expert-tile geometry differs from the record layout");
    }

    // [1,128,128] F3: 12 chunks, 19968 bytes; [2,128,128] E2M1: 2 x 27648 bytes.
    auto directory = normative_directory();
    directory["objects"].push_back({{"name", "f3_records"},
                                    {"kind", "tensor"},
                                    {"shape", {1, 128, 128}},
                                    {"format", "F3G32_E4M3FN"},
                                    {"layout", "moe-expert-tiles-v1"},
                                    {"offset", 3584},
                                    {"bytes", 19968}});
    directory["objects"].push_back({{"name", "e2m1_records"},
                                    {"kind", "tensor"},
                                    {"shape", {2, 128, 128}},
                                    {"format", "E2M1G16_E4M3FN"},
                                    {"layout", "moe-expert-tiles-v1"},
                                    {"offset", 23552},
                                    {"bytes", 55296}});
    auto fixture = write_fixture(directory, "expert-tiles");
    Reader reader(fixture.path);
    const auto* f3_tensor   = std::get_if<TensorDescriptor>(reader.find("f3_records"));
    const auto* e2m1_tensor = std::get_if<TensorDescriptor>(reader.find("e2m1_records"));
    if (f3_tensor == nullptr || f3_tensor->format != NumericFormat::F3G32_E4M3FN ||
        f3_tensor->layout != StorageLayout::MoeExpertTilesV1 ||
        reader.payload(*f3_tensor).data.size() != 19968 || e2m1_tensor == nullptr ||
        e2m1_tensor->format != NumericFormat::E2M1G16_E4M3FN ||
        reader.payload(*e2m1_tensor).data.size() != 55296) {
        throw std::runtime_error("expert-tile directory/payload did not round trip");
    }

    const std::array<std::uint64_t, 2> matrix{128, 128};
    const std::array<std::uint64_t, 3> ragged{1, 128, 160};
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::tensor_encoded_size(StorageLayout::RowSplitK128V1,
                                                        NumericFormat::F3G32_E4M3FN, matrix);
        },
        "F3 cannot use grouped integer layout");
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::tensor_encoded_size(StorageLayout::MoeExpertTilesV1,
                                                        NumericFormat::NVFP4, flash_next);
        },
        "NVFP4 cannot use expert tiles");
    expect_artifact_error(
        [&] {
            (void)ninfer::artifact::tensor_encoded_size(StorageLayout::MoeExpertTilesV1,
                                                        NumericFormat::E2M1G16_E4M3FN, ragged);
        },
        "expert tiles require 128-multiple widths");
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 8 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 8> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4", "q5", "q6", "w8",
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
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1) {
        throw std::runtime_error("fixture object signature mismatch");
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
            if (std::string_view(error.what()).find("NInfer artifact v1 is not supported") ==
                std::string_view::npos) {
                throw std::runtime_error("v1 rejection has the wrong message");
            }
            return;
        }
        throw std::runtime_error("v1 artifact was accepted");
    }
}

std::string all_tensor_identity(const Reader& reader) {
    std::vector<ninfer::artifact::TensorContentSelection> selections;
    for (const auto& object : reader.objects()) {
        if (const auto* tensor = std::get_if<TensorDescriptor>(&object)) {
            selections.push_back({tensor->name, std::nullopt});
        }
    }
    return reader.tensor_content_identity(selections);
}

void test_selected_row_identity() {
    struct Case {
        NumericFormat format;
        StorageLayout layout;
        std::uint64_t rows;
        ninfer::artifact::TensorRowRange selected;
        const char* sha256;
    };

    // Independent Python hashlib vectors use the specified code/scale/divisor ranges and
    // the fixture's 0x01 payload. They verify the exact encoded selection, not just sensitivity.
    const std::array cases{
        Case{NumericFormat::NVFP4,
             StorageLayout::BlockScaleK16M128x4V1,
             384,
             {128, 128},
             "ca65173f34e86215f12c5a0dfbdbc37e076d86e7b0bd3366e5dc5d9721534b2c"},
        Case{NumericFormat::W8G32_F16S,
             StorageLayout::RowSplitK128V1,
             4,
             {1, 2},
             "78f0242f53d877f95142ad09e1cedf9888b953a1570bd7c8629430c667e2217e"},
        Case{NumericFormat::Q4G64_F16S,
             StorageLayout::RowSplitK128V1,
             4,
             {1, 2},
             "193a0b3857803a4aa13da1426b077546157a829b2ba2b0aa83680c64b0b383c1"},
    };
    for (const auto& test : cases) {
        const std::array<std::uint64_t, 2> shape{test.rows, 128};
        const Json directory = {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({{{"name", "weight"},
                                      {"kind", "tensor"},
                                      {"shape", shape},
                                      {"format", ninfer::artifact::format_name(test.format)},
                                      {"layout", ninfer::artifact::layout_name(test.layout)},
                                      {"offset", 0},
                                      {"bytes", ninfer::artifact::tensor_encoded_size(
                                                    test.layout, test.format, shape)}}})},
        };
        auto fixture = write_fixture(directory, "selected-row-identity");
        const Reader reader(fixture.path);
        const std::array selections{
            ninfer::artifact::TensorContentSelection{"weight", test.selected},
            ninfer::artifact::TensorContentSelection{"weight", test.selected},
        };
        if (reader.tensor_content_identity(selections) !=
            std::string("tensor-sha256-v1:") + test.sha256) {
            throw std::runtime_error("selected row SHA-256 disagrees with independent oracle");
        }
    }
}

void test_tensor_content_identity() {
    const Json single = {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({{{"name", "weight"},
                                  {"kind", "tensor"},
                                  {"shape", {2, 3}},
                                  {"format", "BF16"},
                                  {"layout", "contiguous-le-v1"},
                                  {"offset", 0},
                                  {"bytes", 12}}})},
    };
    auto known = write_fixture(single, "tensor-identity-known");
    // Python hashlib over the canonical descriptor including [[0,12]], then twelve 0x01 bytes.
    if (all_tensor_identity(Reader(known.path)) !=
        "tensor-sha256-v1:6ce24c5548cd0703f4ddb6a8d7bb42c7f90e6d5a4b84404f42e24c0e25ec2179") {
        throw std::runtime_error("tensor content SHA-256 disagrees with independent oracle");
    }
    auto original      = write_fixture(normative_directory(), "tensor-identity-original");
    auto replacement   = write_fixture(normative_directory(), "tensor-identity-replacement");
    const auto initial = all_tensor_identity(Reader(original.path));
    if (initial.empty() || all_tensor_identity(Reader(replacement.path)) != initial) {
        throw std::runtime_error("identical tensors on different inodes changed identity");
    }
    std::filesystem::rename(replacement.path, original.path);
    const auto before = std::filesystem::last_write_time(original.path);
    std::filesystem::last_write_time(original.path, before - std::chrono::seconds(2));
    if (all_tensor_identity(Reader(original.path)) != initial) {
        throw std::runtime_error("file replacement or timestamp changed tensor identity");
    }
    const auto overwrite = [&](std::uint64_t offset, char byte) {
        std::fstream file(original.path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(offset));
        file.put(byte);
        if (!file) { throw std::runtime_error("failed to modify artifact fixture"); }
    };
    const auto payload_offset = Reader(original.path).payload_offset();
    overwrite(payload_offset, 'Z');       // Frontend resource bytes are not numerical model state.
    overwrite(payload_offset + 128, 'P'); // Inter-object padding is not tensor content.
    if (all_tensor_identity(Reader(original.path)) != initial) {
        throw std::runtime_error("resource or padding changed tensor identity");
    }

    auto moved_directory = normative_directory();
    auto& moved_objects  = moved_directory["objects"];
    std::reverse(moved_objects.begin(), moved_objects.end());
    for (std::size_t i = 0; i < moved_objects.size(); ++i) {
        moved_objects[i]["offset"] = i * std::uint64_t{4096};
    }
    auto moved = write_fixture(moved_directory, "tensor-identity-repacked");
    {
        Reader source(original.path);
        Reader destination(moved.path);
        std::fstream file(moved.path, std::ios::in | std::ios::out | std::ios::binary);
        for (const auto& object : destination.objects()) {
            const auto data = source.payload(ninfer::artifact::object_name(object)).data;
            file.seekp(static_cast<std::streamoff>(destination.payload(object).absolute_offset));
            file.write(reinterpret_cast<const char*>(data.data()),
                       static_cast<std::streamsize>(data.size()));
        }
        if (!file) { throw std::runtime_error("failed to repack artifact tensors"); }
    }
    if (all_tensor_identity(Reader(moved.path)) != initial) {
        throw std::runtime_error("object placement or order changed tensor identity");
    }
    overwrite(payload_offset + 256, 'W');
    std::filesystem::last_write_time(original.path, before);
    if (all_tensor_identity(Reader(original.path)) == initial) {
        throw std::runtime_error("changed tensor bytes with restored mtime were not detected");
    }
    for (const auto field : {"shape", "format", "name"}) {
        auto directory = normative_directory();
        auto& tensor   = directory["objects"][1];
        if (std::string_view(field) == "shape") {
            tensor[field] = {3, 2};
        } else if (std::string_view(field) == "format") {
            tensor[field]   = "I32";
            tensor["shape"] = {3};
        } else {
            tensor[field] = "renamed";
        }
        auto changed = write_fixture(directory, std::string("tensor-identity-") + field);
        if (all_tensor_identity(Reader(changed.path)) == initial) {
            throw std::runtime_error(std::string("changed tensor ") + field + " was not detected");
        }
    }
}

// Direct-stream tests. The streamed object is 300,000 bytes of position-dependent data read
// through four 16 KiB buffers, so every range crosses many buffer recycles and unaligned edges.
constexpr std::size_t kStreamBuffer = 16384;
constexpr std::size_t kStreamBytes  = 300000;

std::uint8_t pattern(std::uint64_t index) { return static_cast<std::uint8_t>(index * 131 + 7); }

struct StreamFixture {
    ninfer::test::artifact_fixture::TemporaryArtifact artifact;
    std::unique_ptr<Reader> reader;
    std::uint64_t base = 0; // absolute file offset of the streamed object

    StreamFixture() : artifact(write_stream_artifact()) {
        reader = std::make_unique<Reader>(artifact.path);
        base   = reader->payload("stream").absolute_offset;
        std::fstream file(artifact.path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(base));
        for (std::uint64_t i = 0; i < kStreamBytes; ++i) {
            file.put(static_cast<char>(pattern(i)));
        }
        if (!file) { throw std::runtime_error("failed to fill stream fixture"); }
    }

    static ninfer::test::artifact_fixture::TemporaryArtifact write_stream_artifact() {
        const Json directory = {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({{{"name", "stream"},
                                      {"kind", "resource"},
                                      {"encoding", "raw-bytes-v1"},
                                      {"offset", 0},
                                      {"bytes", kStreamBytes}}})},
        };
        return write_fixture(directory, "stream");
    }
};

struct FreeDeleter {
    void operator()(std::byte* pointer) const noexcept { std::free(pointer); }
};

struct AlignedBuffers {
    std::array<std::unique_ptr<std::byte, FreeDeleter>, Reader::stream_buffer_count> owners;
    Reader::StreamBuffers views;

    explicit AlignedBuffers(std::size_t bytes = kStreamBuffer, std::size_t misalign = 0) {
        for (std::size_t i = 0; i < owners.size(); ++i) {
            owners[i].reset(static_cast<std::byte*>(std::aligned_alloc(
                Reader::direct_io_alignment, bytes + Reader::direct_io_alignment)));
            views[i] = std::span(owners[i].get() + misalign, bytes);
        }
    }
};

// Streams [offset, offset + bytes) of the object and checks the delivered bytes, their order and
// that a buffer is never handed to the reader before its previous completion ran.
void check_stream(const StreamFixture& fixture, std::uint64_t offset, std::uint64_t bytes,
                  std::size_t unit) {
    AlignedBuffers buffers;
    std::array<bool, Reader::stream_buffer_count> busy{};
    std::uint64_t expected = fixture.base + offset;
    std::size_t chunks = 0, completions = 0;
    const auto physical = fixture.reader->stream(
        {fixture.base + offset, bytes}, buffers.views,
        [&](ninfer::artifact::PayloadSpan chunk, std::size_t slot) -> Reader::Completion {
            if (busy.at(slot) || chunk.absolute_offset != expected || chunk.data.empty() ||
                chunk.data.size() % unit != 0 || chunk.data.size() > kStreamBuffer) {
                throw std::runtime_error("stream chunk order, size or buffer reuse is wrong");
            }
            for (std::size_t i = 0; i < chunk.data.size(); ++i) {
                if (std::to_integer<std::uint8_t>(chunk.data[i]) !=
                    pattern(expected - fixture.base + i)) {
                    throw std::runtime_error("stream chunk bytes differ from the file");
                }
            }
            expected += chunk.data.size();
            busy.at(slot) = true;
            ++chunks;
            return [&, slot] {
                busy.at(slot) = false;
                ++completions;
            };
        },
        unit);
    if (expected != fixture.base + offset + bytes || completions != chunks ||
        std::ranges::any_of(busy, [](bool b) { return b; }) || (bytes != 0 && chunks < 2) ||
        physical < bytes) {
        throw std::runtime_error("stream did not deliver the whole range and join every buffer");
    }
}

void test_direct_stream() {
    const StreamFixture fixture;
    check_stream(fixture, 0, kStreamBytes, 1);
    check_stream(fixture, 13, kStreamBytes - 13 - 37, 1);
    check_stream(fixture, 4096 - 1, 70000, 1);
    check_stream(fixture, 4096 + 5, 60 * 1000, 1000);
    check_stream(fixture, 777, 100 * 1500, 1500);
    check_stream(fixture, 100, 0, 1);

    // A consumer that throws, and a completion that throws, still join every earlier completion.
    for (const bool throw_in_completion : {false, true}) {
        AlignedBuffers buffers;
        std::size_t consumed = 0, joined = 0;
        expect_artifact_error(
            [&] {
                try {
                    (void)fixture.reader->stream(
                        {fixture.base, kStreamBytes}, buffers.views,
                        [&](ninfer::artifact::PayloadSpan, std::size_t) -> Reader::Completion {
                            if (++consumed == 6 && !throw_in_completion) {
                                throw ninfer::artifact::ArtifactError("consumer failed");
                            }
                            return [&, index = consumed] {
                                ++joined;
                                if (throw_in_completion && index == 3) {
                                    throw ninfer::artifact::ArtifactError("completion failed");
                                }
                            };
                        });
                } catch (...) {
                    if (joined != consumed - (throw_in_completion ? 0 : 1)) {
                        throw std::runtime_error("failed stream left a completion unjoined");
                    }
                    throw;
                }
            },
            "consumer or completion failure");
    }

    // The consumer contract for asynchronous work: a consumer that throws after starting work
    // for its own chunk settles that work itself and returns no completion, so the stream never
    // calls a completion for that slot, and no work is still running when stream() unwinds.
    {
        AlignedBuffers buffers;
        std::vector<std::future<void>> running;
        std::size_t consumed = 0, completions_run = 0;
        bool threw = false;
        try {
            (void)fixture.reader->stream(
                {fixture.base, kStreamBytes}, buffers.views,
                [&](ninfer::artifact::PayloadSpan, std::size_t) -> Reader::Completion {
                    running.push_back(std::async(std::launch::async, [] {}));
                    if (++consumed == 7) {
                        running.back().get(); // settles its own work before throwing
                        throw ninfer::artifact::ArtifactError("consumer failed after enqueue");
                    }
                    return [&, index = running.size() - 1] {
                        running[index].get();
                        ++completions_run;
                    };
                });
        } catch (const ninfer::artifact::ArtifactError&) { threw = true; }
        const bool all_settled =
            std::ranges::none_of(running, [](const std::future<void>& f) { return f.valid(); });
        if (!threw || !all_settled || completions_run != consumed - 1) {
            throw std::runtime_error("failed consumer left asynchronous work or ran a completion");
        }
    }

    // Invalid buffers, ranges and units.
    const auto noop = [](ninfer::artifact::PayloadSpan, std::size_t) {
        return Reader::Completion{};
    };
    AlignedBuffers good;
    AlignedBuffers misaligned(kStreamBuffer, 1);
    expect_artifact_error(
        [&] { (void)fixture.reader->stream({fixture.base, 100}, misaligned.views, noop); },
        "misaligned buffer");
    auto overlapping = good.views;
    overlapping[1]   = overlapping[0];
    expect_artifact_error(
        [&] { (void)fixture.reader->stream({fixture.base, 100}, overlapping, noop); },
        "overlapping buffers");
    expect_artifact_error(
        [&] { (void)fixture.reader->stream({fixture.base, 1000}, good.views, noop, 300); },
        "range not a whole number of units");
    expect_artifact_error(
        [&] { (void)fixture.reader->stream({fixture.base, 100}, good.views, nullptr); },
        "missing consumer");
    expect_artifact_error(
        [&] {
            (void)fixture.reader->stream({fixture.reader->file_bytes() - 10, 11}, good.views, noop);
        },
        "range past the end of the file");

    // Exact edges through read_bytes.
    std::array<std::byte, 5000> window{};
    (void)fixture.reader->read_bytes(fixture.base + 4093, window);
    for (std::size_t i = 0; i < window.size(); ++i) {
        if (std::to_integer<std::uint8_t>(window[i]) != pattern(4093 + i)) {
            throw std::runtime_error("read_bytes returned the wrong bytes");
        }
    }
    expect_artifact_error(
        [&] { (void)fixture.reader->read_bytes(fixture.reader->file_bytes() - 4, window); },
        "read_bytes past the end");
}

// The file shrinks after the Reader recorded its size: the direct read ends early.
void test_direct_stream_short_read() {
    const StreamFixture fixture;
    AlignedBuffers buffers;
    std::filesystem::resize_file(fixture.artifact.path, fixture.base + 100000);
    expect_artifact_error(
        [&] {
            (void)fixture.reader->stream(
                {fixture.base, kStreamBytes}, buffers.views,
                [](ninfer::artifact::PayloadSpan, std::size_t) { return Reader::Completion{}; });
        },
        "short direct read");
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_fp8_row_layout();
        test_expert_tiles_layout();
        test_normative_fixture();
        test_common_validation();
        test_selected_row_identity();
        test_tensor_content_identity();
        test_direct_stream();
        test_direct_stream_short_read();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
