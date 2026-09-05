#pragma once

#include <nvtx3/nvToolsExt.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen4::verifier::profile {

enum class Phase : std::size_t {
    Token,
    Prefill,
    Layer,
    Qsa,
    SparseMoe,
    Count,
};

[[nodiscard]] inline nvtxDomainHandle_t domain() noexcept {
    static nvtxDomainHandle_t handle = [] {
        nvtxDomainHandle_t out = nvtxDomainCreateA("ninfer.qwen4.verifier");
        nvtxDomainNameCategoryA(out, 1, "token");
        nvtxDomainNameCategoryA(out, 2, "prefill");
        nvtxDomainNameCategoryA(out, 3, "layer");
        nvtxDomainNameCategoryA(out, 4, "qsa");
        nvtxDomainNameCategoryA(out, 5, "sparse-moe");
        return out;
    }();
    return handle;
}

[[nodiscard]] inline nvtxStringHandle_t message(Phase phase) noexcept {
    static constexpr std::array<const char*, static_cast<std::size_t>(Phase::Count)> names{
        "token",
        "prefill",
        "layer",
        "qsa",
        "sparse_moe",
    };
    static const auto handles = [] {
        std::array<nvtxStringHandle_t, names.size()> out{};
        for (std::size_t index = 0; index < names.size(); ++index) {
            out[index] = nvtxDomainRegisterStringA(domain(), names[index]);
        }
        return out;
    }();
    return handles[static_cast<std::size_t>(phase)];
}

[[nodiscard]] inline std::uint32_t color(Phase phase) noexcept {
    static constexpr std::array<std::uint32_t, static_cast<std::size_t>(Phase::Count)> colors{
        0xfff28e2bu,
        0xff59a14fu,
        0xff76b7b2u,
        0xffe15759u,
        0xffb07aa1u,
    };
    return colors[static_cast<std::size_t>(phase)];
}

class ScopedRange {
public:
    explicit ScopedRange(Phase phase, std::uint64_t payload) noexcept : domain_(domain()) {
        nvtxEventAttributes_t attributes{};
        attributes.version            = NVTX_VERSION;
        attributes.size               = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.category           = static_cast<std::uint32_t>(phase) + 1;
        attributes.colorType          = NVTX_COLOR_ARGB;
        attributes.color              = color(phase);
        attributes.payloadType        = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        attributes.payload.ullValue   = payload;
        attributes.messageType        = NVTX_MESSAGE_TYPE_REGISTERED;
        attributes.message.registered = message(phase);
        nvtxDomainRangePushEx(domain_, &attributes);
    }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
    ScopedRange(ScopedRange&&) = delete;
    ScopedRange& operator=(ScopedRange&&) = delete;

    ~ScopedRange() noexcept { nvtxDomainRangePop(domain_); }

private:
    nvtxDomainHandle_t domain_;
};

} // namespace ninfer::targets::qwen4::verifier::profile
