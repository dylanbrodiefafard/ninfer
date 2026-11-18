#include "runtime/expert_helper_device.h"

#include "ninfer/ops/moe_helper_experts.h"

#include <map>
#include <set>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next {
std::string_view expert_helper_toolchain() noexcept { return __clang_version__; }

struct ExpertHelperExecution::Impl {
    const HipStream& stream;
    std::uint32_t capacity;
    HipBuffer x, ids, weights, routes, partial, count;
    ops::MoeHelperExpertsWorkspace workspace;
    HipEvent completion;
    bool pending = false;
    std::map<std::uint32_t, std::unique_ptr<HipGraph>> graphs;

    Impl(ExpertMailboxView mailbox, const std::array<ExpertHelperBank, 48>& banks,
         std::int32_t lanes, bool speculative, const HipStream& queue)
        : stream(queue), capacity(static_cast<std::uint32_t>(lanes * (speculative ? 4 : 1))),
          x(capacity * kExpertActivationBytes), ids(static_cast<std::size_t>(capacity) * 40),
          weights(static_cast<std::size_t>(capacity) * 40),
          routes(static_cast<std::size_t>(capacity) * 80), partial(capacity * kExpertPartialBytes),
          count(sizeof(std::int32_t)), workspace(static_cast<std::int32_t>(capacity)) {
        for (auto* p : {&x, &ids, &weights, &routes, &partial, &count}) {
            hip_zero(p->data(), p->size(), stream);
        }
        stream.synchronize();
        std::set<std::uint32_t> shapes;
        for (std::int32_t width = 1; width <= (speculative ? 4 : 1); ++width) {
            for (std::int32_t batch = 1; batch <= lanes; ++batch) {
                shapes.insert(static_cast<std::uint32_t>(batch * width));
            }
        }
        for (const auto t : shapes) {
            auto graph = std::make_unique<HipGraph>(stream);
            for (std::uint32_t layer = 0; layer < 48; ++layer) {
                expert_mailbox_hip_import(mailbox, layer, x.data(), ids.data(), weights.data(),
                                          static_cast<std::uint32_t*>(count.data()), t, stream);
                expert_mailbox_hip_interleave(ids.data(), weights.data(), routes.data(),
                                              static_cast<std::uint32_t*>(count.data()), stream);
                const auto& bank = banks[layer];
                ops::moe_helper_experts(
                    ops::MoeHelperExpertsArgs{
                        .tokens     = static_cast<const std::int32_t*>(count.data()),
                        .max_tokens = static_cast<std::int32_t>(t),
                        .x          = static_cast<const std::uint16_t*>(x.data()),
                        .routes     = static_cast<const ops::MoeHelperRoute*>(routes.data()),
                        .bank       = {.expert_table  = bank.records,
                                       .matrix_scales = bank.matrix_scales,
                                       .f3_books      = bank.f3_books},
                        .y          = static_cast<float*>(partial.data())},
                    workspace, static_cast<hipStream_t>(stream.get()));
                expert_mailbox_hip_export(mailbox, layer, partial.data(),
                                          static_cast<std::uint32_t*>(count.data()), t, stream);
            }
            graph->instantiate();
            graphs.emplace(t, std::move(graph));
        }
    }
};

ExpertHelperExecution::ExpertHelperExecution(ExpertMailboxView mailbox,
                                             const std::array<ExpertHelperBank, 48>& banks,
                                             std::int32_t lanes, bool speculative,
                                             const HipStream& stream) {
    if (lanes < 1 || lanes > 12 ||
        mailbox.capacity != static_cast<std::uint32_t>(lanes * (speculative ? 4 : 1))) {
        throw std::invalid_argument("expert helper graph capacity/lanes");
    }
    impl_ = std::make_unique<Impl>(mailbox, banks, lanes, speculative, stream);
}

ExpertHelperExecution::~ExpertHelperExecution() {
    try {
        impl_->stream.synchronize();
    } catch (...) { std::terminate(); }
}

void ExpertHelperExecution::launch(std::uint32_t tokens) {
    if (impl_->pending) { throw std::logic_error("expert helper graph completion was not joined"); }
    const auto found = impl_->graphs.find(tokens);
    if (found == impl_->graphs.end()) {
        throw std::invalid_argument("expert helper round shape was not instantiated");
    }
    found->second->launch(impl_->stream);
    impl_->completion.record(impl_->stream);
    impl_->pending = true;
}

void ExpertHelperExecution::join() {
    if (impl_->pending) {
        impl_->completion.synchronize();
        impl_->pending = false;
    }
}
} // namespace ninfer::targets::qwen3_8_flash_next
