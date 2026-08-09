#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

template <class Body>
void run_prepared(State& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        executable->launch(state.device.stream);
    } else {
        body();
    }
}

template <class Body>
void warm_capture(State& state, DecodeGraphDefinition& definition, const GraphPrepare& prepare,
                  Body&& body) {
    prepare();
    state.device.synchronize();
    body();
    state.device.synchronize();
    prepare();
    state.device.synchronize();
    definition.capture(state.device.stream, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
