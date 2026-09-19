#pragma once

#include <text/qwen/frontend.h>
#include <text/qwen/frontend_resources.h>
#include <text/qwen/prepared_prompt.h>

namespace ninfer::text::qwen {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled = true,
                                                   FrontendProfile profile = FrontendProfile::Qwen3_6);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace ninfer::text::qwen
