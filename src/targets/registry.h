#pragma once

#include "ninfer/types.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <cstdint>
#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;
class HostPinnedArena;

namespace targets {

using Qwen3_6_27B = qwen3_6_27b::Package;

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    explicit LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_6_27B::SequencePlan sequence_plan, DeviceContext& device,
                        std::unique_ptr<HostPinnedArena> kv_ram_arena);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

using Qwen3_8FlashNext = qwen3_8_flash_next::Package;

struct LoadedQwen3_8FlashNext {
    std::unique_ptr<Qwen3_8FlashNext::LoadedModel> model;
    Qwen3_8FlashNext::Frontend frontend;

    explicit LoadedQwen3_8FlashNext(std::unique_ptr<Qwen3_8FlashNext::LoadedModel> stable_model);
    ~LoadedQwen3_8FlashNext();

    LoadedQwen3_8FlashNext(const LoadedQwen3_8FlashNext&)            = delete;
    LoadedQwen3_8FlashNext& operator=(const LoadedQwen3_8FlashNext&) = delete;
};

// The executor Instance of qwen3.8-flash-next. Its Program owns no request transient, so
// request_memory is an empty frozen region; capacity is the Engine's max_context.
struct Qwen3_8FlashNextInstance {
    using Package = Qwen3_8FlashNext;

    std::unique_ptr<LoadedQwen3_8FlashNext> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_8FlashNext::Program> program;

    Qwen3_8FlashNextInstance(std::unique_ptr<LoadedQwen3_8FlashNext> stable_loaded,
                             const EngineOptions& options, DeviceContext& device);
    ~Qwen3_8FlashNextInstance();

    Qwen3_8FlashNextInstance(const Qwen3_8FlashNextInstance&)            = delete;
    Qwen3_8FlashNextInstance& operator=(const Qwen3_8FlashNextInstance&) = delete;
};

// The registered target the Engine runs; the Engine dispatches on it with std::visit.
using ActiveTarget =
    std::variant<std::unique_ptr<Qwen3_6_27BInstance>, std::unique_ptr<Qwen3_8FlashNextInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer
