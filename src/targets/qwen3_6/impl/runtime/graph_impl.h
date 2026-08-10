#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        executable->launch(state.execution.device.stream);
    } else {
        body();
    }
}

template <class Context, class Body>
void warm_capture(Context& state, DecodeGraphDefinition& definition, const GraphPrepare& prepare,
                  Body&& body) {
    prepare();
    state.execution.device.synchronize();
    body();
    state.execution.device.synchronize();
    prepare();
    state.execution.device.synchronize();
    definition.capture(state.execution.device.stream, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
