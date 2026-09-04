#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/dtype.h"
#include "product/prompt_input/prompt_input.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#undef NINFER_QWEN36_RUNTIME_NS
#undef NINFER_QWEN36_VARIANT

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Json = nlohmann::json;
using ninfer::targets::qwen3_6_27b::detail::ArtifactLoadPlan;
using ninfer::targets::qwen3_6_27b::detail::LoadedModelData;
using ninfer::targets::qwen3_6_27b::detail::Variant;
using ninfer::targets::qwen3_6_27b::detail::WeightsProfile;
namespace runtime = ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime;

const char* dtype_name(ninfer::DType dtype) {
    switch (dtype) {
    case ninfer::DType::BF16:
        return "bf16";
    case ninfer::DType::FP32:
        return "fp32";
    case ninfer::DType::FP16:
        return "fp16";
    case ninfer::DType::I32:
        return "i32";
    default:
        throw std::invalid_argument("Vision trace does not support this tensor dtype");
    }
}

std::vector<std::int32_t> python_shape(const ninfer::Tensor& value) {
    int rank = 4;
    while (rank > 1 && value.ne[rank - 1] == 1) { --rank; }
    std::vector<std::int32_t> shape;
    shape.reserve(static_cast<std::size_t>(rank));
    for (int dim = rank - 1; dim >= 0; --dim) { shape.push_back(value.ne[dim]); }
    return shape;
}

std::vector<std::byte> copy_logical(const ninfer::Tensor& value, cudaStream_t stream) {
    const std::size_t element = ninfer::dtype_size(value.dtype);
    std::vector<std::byte> host(static_cast<std::size_t>(value.numel()) * element);
    if (value.is_contiguous()) {
        CUDA_CHECK(cudaMemcpyAsync(host.data(), value.data, host.size(), cudaMemcpyDeviceToHost,
                                   stream));
    } else if (value.ne[3] == 1 && value.nb[0] == static_cast<std::int64_t>(element) &&
               value.nb[1] == value.nb[0] * value.ne[0]) {
        const std::size_t row = static_cast<std::size_t>(value.ne[0]) *
                                static_cast<std::size_t>(value.ne[1]) * element;
        CUDA_CHECK(cudaMemcpy2DAsync(host.data(), row, value.data,
                                     static_cast<std::size_t>(value.nb[2]), row,
                                     static_cast<std::size_t>(value.ne[2]),
                                     cudaMemcpyDeviceToHost, stream));
    } else {
        throw std::invalid_argument("Vision trace encountered an unsupported strided tensor");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}

void write_bytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) { throw std::runtime_error("failed to write Vision trace tensor: " + path.string()); }
}

class FileTrace final : public runtime::schedule::VisionTraceSink {
public:
    FileTrace(std::filesystem::path root, Json& records, std::size_t item)
        : root_(std::move(root)), records_(records), item_(item) {}

    void capture(std::string_view name, const ninfer::Tensor& value,
                 cudaStream_t stream) override {
        const std::string relative = "item_" + (item_ < 10 ? std::string("0") : "") +
                                     std::to_string(item_) + "/" + std::string(name) + ".bin";
        const std::vector<std::byte> bytes = copy_logical(value, stream);
        write_bytes(root_ / relative, bytes);
        records_.push_back({{"item", item_},
                            {"name", name},
                            {"file", relative},
                            {"dtype", dtype_name(value.dtype)},
                            {"shape", python_shape(value)}});
    }

private:
    std::filesystem::path root_;
    Json& records_;
    std::size_t item_;
};

Json item_json(const ninfer::targets::qwen3_6::VisionItemControl& item) {
    return {{"modality", item.modality == ninfer::targets::qwen3_6::PromptModality::Image
                              ? "image"
                              : "video"},
            {"grid", {item.grid.temporal, item.grid.height, item.grid.width}},
            {"patch_begin", item.patch_begin},
            {"patch_count", item.patch_count},
            {"merged_count", item.merged_count},
            {"segment_length", item.segment_length},
            {"segment_count", item.segment_count},
            {"position_ids", item.position_ids},
            {"cu_seqlens", item.cu_seqlens},
            {"scatter_indices", item.scatter_indices},
            {"position_table_indices", item.position_table_indices},
            {"position_table_weights", item.position_table_weights}};
}

ninfer::targets::qwen3_6::VisionItemControl aggregate_controls(
    std::span<const ninfer::targets::qwen3_6::VisionItemControl> items) {
    if (items.empty()) { throw std::invalid_argument("cannot aggregate an empty Vision item list"); }
    ninfer::targets::qwen3_6::VisionItemControl out;
    out.modality    = items.front().modality;
    out.patch_begin = items.front().patch_begin;
    out.cu_seqlens.push_back(0);
    std::int32_t patch_base = 0;
    for (int axis = 0; axis < 2; ++axis) {
        for (const auto& item : items) {
            const auto begin = item.position_ids.begin() +
                               static_cast<std::ptrdiff_t>(axis * item.patch_count);
            out.position_ids.insert(out.position_ids.end(), begin,
                                    begin + static_cast<std::ptrdiff_t>(item.patch_count));
        }
    }
    for (const auto& item : items) {
        if (item.patch_begin != out.patch_begin + out.patch_count) {
            throw std::invalid_argument("Vision trace aggregate patch ranges are not contiguous");
        }
        for (std::size_t segment = 1; segment < item.cu_seqlens.size(); ++segment) {
            out.cu_seqlens.push_back(patch_base + item.cu_seqlens[segment]);
        }
        patch_base += static_cast<std::int32_t>(item.patch_count);
        out.patch_count += item.patch_count;
        out.merged_count += item.merged_count;
        out.segment_count += item.segment_count;
        out.position_table_indices.insert(out.position_table_indices.end(),
                                          item.position_table_indices.begin(),
                                          item.position_table_indices.end());
        out.position_table_weights.insert(out.position_table_weights.end(),
                                          item.position_table_weights.begin(),
                                          item.position_table_weights.end());
    }
    return out;
}

int run(const std::filesystem::path& artifact_path, const std::filesystem::path& messages_path,
        const std::filesystem::path& output_root) {
    if (std::filesystem::exists(output_root / "manifest.json")) {
        throw std::invalid_argument("Vision trace output already contains manifest.json");
    }
    std::filesystem::create_directories(output_root);

    ninfer::PromptInput input =
        ninfer::product::prompt_from_messages(messages_path, false, true);
    ninfer::DeviceContext device(0);
    ninfer::artifact::Reader reader(artifact_path);
    const WeightsProfile profile =
        ninfer::targets::qwen3_6_27b::Package::resolve_weights(reader.identity());
    ninfer::artifact::Binder binder(reader);
    ArtifactLoadPlan load = ninfer::targets::qwen3_6_27b::detail::bind_artifact(
        binder, profile, {.vision = true});
    ninfer::artifact::MaterializedArtifact materialized =
        ninfer::artifact::materialize(reader, load.materialization, device);
    LoadedModelData model(std::move(load.bindings), std::move(materialized));
    auto frontend = ninfer::targets::qwen3_6::make_frontend(model.frontend, true);
    auto prepared = frontend.prepare(std::move(input));
    auto prompt = ninfer::targets::qwen3_6::PreparedPromptAccess::take(std::move(prepared));
    const ninfer::targets::qwen3_6::VisionControl control =
        ninfer::targets::qwen3_6::build_vision_control(prompt);
    if (control.items.empty()) { throw std::invalid_argument("Vision trace prompt has no media"); }

    Json manifest{{"format", "ninfer_vision_intermediate_trace_v1"},
                  {"artifact", std::filesystem::absolute(artifact_path).string()},
                  {"messages", std::filesystem::absolute(messages_path).string()},
                  {"token_ids", prompt.token_ids},
                  {"token_types", prompt.token_types},
                  {"positions", prompt.positions},
                  {"rope_delta", prompt.rope_delta},
                  {"items", Json::array()},
                  {"aggregate_matches_individual", control.items.size() < 2},
                  {"captures", Json::array()}};

    runtime::schedule::VisionContext context(device, model.runtime);
    std::vector<std::byte> individual_outputs;
    for (std::size_t index = 0; index < control.items.size(); ++index) {
        const auto& item = control.items[index];
        manifest["items"].push_back(item_json(item));
        const std::size_t patch_offset =
            item.patch_begin * static_cast<std::size_t>(runtime::schedule::VisionScheduleConfig::patch_dim);
        const std::size_t patch_elements =
            item.patch_count * static_cast<std::size_t>(runtime::schedule::VisionScheduleConfig::patch_dim);
        if (patch_offset > prompt.patches.size() ||
            patch_elements > prompt.patches.size() - patch_offset) {
            throw std::logic_error("Vision trace item patch range exceeds prepared prompt");
        }
        ninfer::WorkspaceArena workspace(runtime::schedule::VisionContext::workspace_bytes(item));
        ninfer::DeviceBuffer output_buffer(
            runtime::schedule::VisionContext::output_transient_bytes(item.merged_count));
        ninfer::Tensor output(output_buffer.p, ninfer::DType::BF16,
                              {runtime::schedule::VisionScheduleConfig::out_hidden,
                               static_cast<std::int32_t>(item.merged_count)});
        FileTrace trace(output_root, manifest["captures"], index);
        context.encode({std::span<const float>(prompt.patches).subspan(patch_offset, patch_elements),
                        &item},
                       output, workspace, &trace);
        const std::vector<std::byte> traced_output = copy_logical(output, device.stream);
        individual_outputs.insert(individual_outputs.end(), traced_output.begin(),
                                  traced_output.end());

        // The observation sink must not perturb the production arithmetic.
        ninfer::WorkspaceArena plain_workspace(
            runtime::schedule::VisionContext::workspace_bytes(item));
        ninfer::DeviceBuffer plain_output_buffer(
            runtime::schedule::VisionContext::output_transient_bytes(item.merged_count));
        ninfer::Tensor plain_output(plain_output_buffer.p, ninfer::DType::BF16,
                                    {runtime::schedule::VisionScheduleConfig::out_hidden,
                                     static_cast<std::int32_t>(item.merged_count)});
        context.encode({std::span<const float>(prompt.patches).subspan(patch_offset, patch_elements),
                        &item},
                       plain_output, plain_workspace);
        if (copy_logical(plain_output, device.stream) != traced_output) {
            throw std::runtime_error("Vision trace changed the production output");
        }
    }

    if (control.items.size() > 1) {
        const auto aggregate = aggregate_controls(control.items);
        ninfer::WorkspaceArena aggregate_workspace(
            runtime::schedule::VisionContext::workspace_bytes(aggregate));
        ninfer::DeviceBuffer aggregate_output_buffer(
            runtime::schedule::VisionContext::output_transient_bytes(aggregate.merged_count));
        ninfer::Tensor aggregate_output(
            aggregate_output_buffer.p, ninfer::DType::BF16,
            {runtime::schedule::VisionScheduleConfig::out_hidden,
             static_cast<std::int32_t>(aggregate.merged_count)});
        const std::size_t patch_offset =
            aggregate.patch_begin *
            static_cast<std::size_t>(runtime::schedule::VisionScheduleConfig::patch_dim);
        const std::size_t patch_elements =
            aggregate.patch_count *
            static_cast<std::size_t>(runtime::schedule::VisionScheduleConfig::patch_dim);
        context.encode({std::span<const float>(prompt.patches).subspan(patch_offset, patch_elements),
                        &aggregate},
                       aggregate_output, aggregate_workspace);
        manifest["aggregate_matches_individual"] =
            copy_logical(aggregate_output, device.stream) == individual_outputs;
        if (!manifest["aggregate_matches_individual"].get<bool>()) {
            throw std::runtime_error("aggregated Vision output differs from per-item execution");
        }
    }

    std::ofstream manifest_file(output_root / "manifest.json", std::ios::trunc);
    manifest_file << manifest.dump(2) << '\n';
    if (!manifest_file) { throw std::runtime_error("failed to write Vision trace manifest"); }
    std::cout << "captured " << manifest["captures"].size() << " intermediates across "
              << control.items.size() << " media items\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: ninfer_qwen3_8_27b_vision_trace ARTIFACT MESSAGES OUTPUT_DIR\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "Vision trace failed: " << error.what() << '\n';
        return 1;
    }
}
